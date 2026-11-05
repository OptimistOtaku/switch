"""Confirming a chat account link by tapping a number (CHOO-4121).

The code a person posts from their chat account (CHOO-3881) proves two things
at once: the chat account acted, and the person signed in to Switch saw the
code. A prompt proves the same two things without the typing. Switch's bot
messages the account a few numbers, Switch Console shows the Switch user which
is right, and the account taps it. A wrong number or `Not me` cancels the
request, so a person can't be talked into confirming a request they didn't
start by guessing.

The prompt goes out with a code and lives as long as it does, on the same
request. The code stays the way on wherever a prompt can't be sent: a platform
whose bot can't open a conversation with this person, a message that failed,
or an account that has had its allowance of prompts for the hour.
"""

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from switch_core.account_links import new_prompt_choices, new_prompt_token
from switch_core.bridges.collaboration.adapter import PlatformAdapter
from switch_core.bridges.collaboration.models import (
    DirectMessageUnavailable,
    DirectPrompt,
    InboundInteraction,
)
from switch_core.bridges.collaboration.session.renderers import (
    parse_answer_position,
)
from switch_core.chat_account_links import linked_notice
from switch_core.db.audit import AuditAction, record_audit_event
from switch_core.db.models import (
    CollaborationBridge,
    ExternalUser,
    ExternalUserLinkRequest,
)
from switch_core.db.session_scope import tenant_session
from switch_core.db.stores.external_user_store import ExternalUserStore
from switch_core.db.stores.user_store import UserStore

logger = logging.getLogger(__name__)

NOT_ME_LABEL = "Not me"

PROMPT_LIMITED = (
    "Switch has already messaged this account several times in the last hour, "
    "so use the code instead."
)
BRIDGE_STOPPED = (
    "The messaging app isn't connected to Switch right now, so use the code instead."
)

DECLINED = (
    "Cancelled. This account wasn't linked to anyone. If you didn't start "
    "this, nothing else needs doing."
)
WRONG_NUMBER = (
    "That isn't the number Switch Console showed, so nothing was linked and "
    "this request is cancelled. If you meant to link this account, start again "
    "in Switch Console."
)
EXPIRED = (
    "This request has expired, so nothing was linked. Start again in Switch "
    "Console if you meant to link this account."
)
NOT_WAITING = (
    "This request is no longer waiting: it was confirmed, cancelled, or "
    "replaced by a newer one. Nothing was changed."
)
NO_LONGER_MEMBER = (
    "The Switch user who started this is no longer in the workspace, so "
    "nothing was linked."
)
SOMEONE_ELSE = "This request is for a different account, so your tap changed nothing."
NOT_AN_OPTION = "That button isn't one this message offered, so nothing changed."


def compose_prompt(
    *,
    requester_name: str,
    requester_email: str,
    platform: str,
    choices: list[int],
    token: str,
) -> DirectPrompt:
    """The message an account is sent to confirm a link.

    Names who started it, by name and address, because that is what tells a
    person who did not start it that they should tap `Not me`.
    """
    text = (
        f"**{requester_name}** ({requester_email}) is linking this {platform} "
        "account to their Switch account.\n\n"
        "Tap the number Switch Console is showing to finish. If you didn't "
        f"start this, tap **{NOT_ME_LABEL}** and nothing is linked."
    )
    return DirectPrompt(
        text=text,
        token=token,
        labels=(*(str(choice) for choice in choices), NOT_ME_LABEL),
    )


@dataclass(frozen=True)
class PendingPrompt:
    """A prompt recorded against a request, waiting for its commit before it is
    sent: sent first, a quick tap could arrive before the token it carries is
    there to be found."""

    number: int
    message: DirectPrompt | None
    # Why nothing will be sent, when `message` is None.
    withheld: str | None


@dataclass(frozen=True)
class PromptOutcome:
    """What Switch Console is told: the number to show, and whether the
    message went out, with why not when it did not."""

    number: int
    sent: bool
    unavailable_reason: str | None


async def prepare_prompt(
    session: AsyncSession,
    *,
    external_user_store: ExternalUserStore,
    request: ExternalUserLinkRequest,
    account: ExternalUser,
    requester_name: str,
    requester_email: str,
    platform: str,
    adapter: PlatformAdapter | None,
) -> PendingPrompt:
    """Record a prompt for the code just issued on `request`, in the caller's
    transaction, unless it cannot be sent.

    The allowance is spent here, before the send, because a send that fails
    has still reached the platform as far as anyone can tell.
    """
    number, choices = new_prompt_choices()
    if adapter is None:
        return PendingPrompt(number=number, message=None, withheld=BRIDGE_STOPPED)
    if not await external_user_store.take_link_prompt_allowance(session, account):
        return PendingPrompt(number=number, message=None, withheld=PROMPT_LIMITED)
    token = new_prompt_token()
    await external_user_store.attach_link_prompt(
        session, request, token=token, number=number, choices=choices
    )
    return PendingPrompt(
        number=number,
        message=compose_prompt(
            requester_name=requester_name,
            requester_email=requester_email,
            platform=platform,
            choices=choices,
            token=token,
        ),
        withheld=None,
    )


async def deliver_prompt(
    pending: PendingPrompt,
    *,
    adapter: PlatformAdapter | None,
    account_external_id: str,
) -> PromptOutcome:
    """Send a prepared prompt, after its transaction has committed. Holds no
    connection, since it waits on the platform."""
    if pending.message is None or adapter is None:
        return PromptOutcome(
            number=pending.number,
            sent=False,
            unavailable_reason=pending.withheld or BRIDGE_STOPPED,
        )
    try:
        await adapter.send_direct_prompt(account_external_id, pending.message)
    except DirectMessageUnavailable as unavailable:
        return PromptOutcome(
            number=pending.number, sent=False, unavailable_reason=str(unavailable)
        )
    except Exception:
        logger.exception(
            "Could not send a link prompt to %s on %s; the code still works.",
            account_external_id,
            adapter.display_name,
        )
        return PromptOutcome(
            number=pending.number,
            sent=False,
            unavailable_reason=(
                f"{adapter.display_name} didn't accept the message, so use the "
                "code instead."
            ),
        )
    return PromptOutcome(number=pending.number, sent=True, unavailable_reason=None)


async def answer_prompt(
    interaction: InboundInteraction,
    *,
    adapter: PlatformAdapter,
    session_factory: async_sessionmaker[AsyncSession],
    tenant_id: str,
    external_user_store: ExternalUserStore,
    user_store: UserStore,
) -> None:
    """Act on a tap on a prompt's buttons, then replace the prompt with what
    happened.

    Who tapped is what the platform says, not anything the button carried, and
    only the account the prompt was sent to can answer it. The request is
    locked while it is decided, so a tap and a code posted at the same moment
    cannot both confirm it, and a tap racing a newer code finds the newer code
    has retired its token.
    """
    position = parse_answer_position(interaction.action_id)
    # What to tell the presser alone, or what to replace the prompt with once
    # the request is settled. Acted on after the session closes, so no
    # connection is held while the platform is asked.
    notice: str | None = None
    outcome: str | None = None
    async with tenant_session(session_factory, tenant_id) as session:
        found = await external_user_store.lock_link_request_by_prompt(
            session, interaction.value
        )
        if found is None:
            # Told rather than written over the message: a second tap racing
            # the first finds the request already confirmed, and replacing
            # "Linked" with this would undo what it showed.
            notice = NOT_WAITING
        else:
            request, account, live = found
            choices = request.prompt_choices or []
            if account.external_user_id != interaction.sender_id:
                notice = SOMEONE_ELSE
            elif position is None or position > len(choices) + 1:
                notice = NOT_AN_OPTION
            else:
                outcome = await _decide(
                    session,
                    request=request,
                    account=account,
                    live=live,
                    pressed=(
                        None if position == len(choices) + 1 else choices[position - 1]
                    ),
                    external_user_store=external_user_store,
                    user_store=user_store,
                )
    if outcome is None:
        await _tell(adapter, interaction, notice or NOT_WAITING)
    elif interaction.message_ref is not None:
        await adapter.settle_direct_prompt(interaction.message_ref, outcome)
    else:
        await _tell(adapter, interaction, outcome)


async def _decide(
    session: AsyncSession,
    *,
    request: ExternalUserLinkRequest,
    account: ExternalUser,
    live: bool,
    pressed: int | None,
    external_user_store: ExternalUserStore,
    user_store: UserStore,
) -> str:
    """Settle a locked request on a tap, commit, and say what happened.
    `pressed` is the number tapped, or None for `Not me`."""
    if not live:
        return EXPIRED
    user_id = request.user_id
    details = {"bridge_id": account.bridge_id, "user_id": user_id, "by": "prompt"}
    if pressed is None or pressed != request.prompt_number:
        await session.delete(request)
        await record_audit_event(
            session,
            tenant_id=account.tenant_id,
            actor_user_id=None,
            action=AuditAction.BRIDGE_IDENTITY_LINK_CANCELLED,
            target_type="external_user",
            target_id=account.id,
            details={
                **details,
                "reason": "declined" if pressed is None else "wrong_number",
            },
        )
        await session.commit()
        return DECLINED if pressed is None else WRONG_NUMBER
    # Standing is checked as a code's redeem checks it: a request outlives a
    # member's removal, and the claim it turns into decides who the account is
    # recognised as.
    if (
        await user_store.effective_tenant_role(session, request.tenant_id, user_id)
        is None
    ):
        await session.delete(request)
        await session.commit()
        return NO_LONGER_MEMBER
    linked = await user_store.get(session, user_id)
    bridge = await session.get(CollaborationBridge, account.bridge_id)
    # Both are held by foreign keys from the request being confirmed.
    if linked is None or bridge is None:
        raise RuntimeError(
            f"Confirming a link for user {user_id} on bridge {account.bridge_id}, "
            "but could not read either"
        )
    name, platform = linked.name, bridge.display_name
    username = account.external_username
    await external_user_store.confirm_link_request(session, request)
    await record_audit_event(
        session,
        tenant_id=account.tenant_id,
        actor_user_id=user_id,
        action=AuditAction.BRIDGE_IDENTITY_CLAIMED,
        target_type="external_user",
        target_id=account.id,
        details=details,
    )
    await session.commit()
    return linked_notice(platform, username, name, ran_command=False)


async def _tell(
    adapter: PlatformAdapter, interaction: InboundInteraction, text: str
) -> None:
    await adapter.tell_actor(
        interaction.channel_id,
        interaction.sender_id,
        interaction.sender_name,
        interaction.thread_ref,
        text,
    )

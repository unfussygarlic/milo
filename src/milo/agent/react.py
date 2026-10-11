"""The ReAct loop: generate -> stop at <|tool_result|> -> run tool -> inject -> continue.

  1. The customer sends a message.
  2. Milo generates until it emits ``<|tool_result|>`` (i.e. it has finished
     writing ``TOOL(args)``).
  3. The runtime parses the call, executes it, injects ``result<|end_tool|>``.
  4. Milo continues - another tool call, or the final answer.

Tools come from :mod:`milo.tools`, the same code the data generator uses, so
results at inference match the results seen in training (invariant 1).

Two guard rails are always on in production (invariant 4):
  * constrained decoding (:class:`milo.agent.constraint.CopyConstraint`) - store /
    search / weather args can only be copied from the customer's message, and
    numbers only from the message or a tool result;
  * the number-grounding guard - if the reply still contains an ungrounded
    number, it is replaced by a deterministic answer built from the tool results.
"""

from milo.agent.constraint import CopyConstraint
from milo.agent.decoding import generate
from milo.model.tokenizer import (
    ASST_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    decode,
    encode,
    strip_tags,
)
from milo.tools import (
    END_TOOL_TAG,
    TOOL_CALL_TAG,
    TOOL_RESULT_TAG,
    execute,
    fallback_answer,
    parse_tool_call,
    ungrounded_numbers,
)


def react_step(model, prompt_tokens: list[int], device: str = "cuda", temperature: float = 0.7,
               max_new_tokens: int = 256, allow_tools: bool = True,
               constraint: CopyConstraint | None = None) -> tuple:
    """
    One step of the ReAct loop.

    allow_tools=True:  generate until <|tool_result|> (the model has finished
                       writing TOOL(args)) or EOS. If a call is found, execute
                       it and return the prompt augmented with the result.
    allow_tools=False: generate a final answer; anything from <|tool_call|>
                       onward is discarded.

    Returns (text, tool_called, tokens, tool_name, args, result)
    """
    stop = [TOOL_RESULT_TAG] if allow_tools else [TOOL_CALL_TAG]
    generated = generate(
        model, prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_strings=stop,
        device=device,
        constraint=constraint,
    )
    new_text = decode(generated[len(prompt_tokens):])

    if allow_tools:
        call = parse_tool_call(new_text)
        if call:
            result = execute(call.name, call.args)
            augmented = (f"{call.before}{TOOL_CALL_TAG}{call.name}({call.args})"
                         f"{TOOL_RESULT_TAG}{result}{END_TOOL_TAG}")
            augmented_tokens = encode(decode(prompt_tokens) + augmented)
            return strip_tags(call.before), True, augmented_tokens, call.name, call.args, result

    # Final response: cut at any (unparseable / disallowed) tool call
    cut = new_text.find(TOOL_CALL_TAG)
    if cut >= 0:
        new_text = new_text[:cut]
    return strip_tags(new_text), False, generated, None, None, None


def build_prompt(user_message: str, conversation_history: list[tuple[str, str]] | None = None) -> str:
    prompt = ""
    for role, content in (conversation_history or []):
        if role == "user":
            prompt += f"{USER_TOKEN}\n{content}\n"
        else:
            prompt += f"{ASST_TOKEN}\n{content}{EOS_TOKEN}\n"
    prompt += f"{USER_TOKEN}\n{user_message}\n{ASST_TOKEN}\n"
    return prompt


def react_loop(model, user_message: str, conversation_history: list[tuple[str, str]] | None = None,
               device: str = "cuda", max_tool_calls: int = 3, temperature: float = 0.7,
               verbose: bool = True, guard: bool = True,
               constrain: bool = True) -> tuple[str, list[tuple[str, str, str]]]:
    """
    Full ReAct loop for a user message.

    guard=True: if tools were called and the final reply contains a number that
    is neither in the user's message nor in any tool result (a mis-copied price,
    an invented total), the reply is replaced by a deterministic one built from
    the tool results. Never let a wrong ₹ amount reach a customer.

    constrain=True: constrained decoding (milo_tools.CopyConstraint) — store /
    search / weather args can only be copied from the customer's message, and
    numbers only from the message or tool results, so the guard rarely fires.

    Returns:
        final_response: The model's complete response text
        tool_calls: List of (tool_name, args, result) tuples
    """
    prompt_tokens = encode(build_prompt(user_message, conversation_history))
    tool_calls = []
    response_parts = []

    for step_i in range(max_tool_calls + 1):
        allow = step_i < max_tool_calls   # last iteration forces a final answer
        constraint = (CopyConstraint(user_message, [user_message] + [r for _, _, r in tool_calls],
                                     numbers_in_reply=bool(tool_calls)) if constrain else None)
        text, tool_called, tokens, tool_name, args, result = react_step(
            model, prompt_tokens, device=device, temperature=temperature,
            allow_tools=allow, constraint=constraint,
        )

        if tool_called:
            if text:
                response_parts.append(text)
            tool_calls.append((tool_name, args, result))
            if verbose:
                print(f"  🔧 {tool_name}({args})")
                print(f"  ⟶  {result}")
            prompt_tokens = tokens
        else:
            response_parts.append(text)
            break

    final_response = " ".join(p for p in response_parts if p).strip()
    if guard and tool_calls:
        bad = ungrounded_numbers(final_response, [user_message] + [r for _, _, r in tool_calls])
        if bad:
            if verbose:
                print(f"  🛡  guard: ungrounded {bad} → answering from tool results")
            final_response = fallback_answer(tool_calls)
    return final_response, tool_calls

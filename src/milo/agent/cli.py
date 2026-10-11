"""``milo-chat`` - interactive terminal chat with the tool-using agent.

Usage:
    milo-chat --checkpoint experiments/tools_v51/checkpoints/best.pt
    milo-chat --checkpoint best.pt --max-tool-calls 3
    MILO_DISABLE_RUN_CODE=1 milo-chat --checkpoint best.pt
    MILO_MERCHANT_KB=merchant_kbs/kirana.json milo-chat --checkpoint best.pt
"""

import argparse
import time

from milo.agent.react import react_loop
from milo.config import generation_config
from milo.model.checkpoint import load_model
from milo.tools import TOOLS


def main():
    parser = argparse.ArgumentParser(description="Milo ReAct Agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to tool-use checkpoint (best.pt)")
    gen = generation_config()
    parser.add_argument("--device", type=str, default=gen["device"])
    parser.add_argument("--temperature", type=float, default=gen["temperature"])
    parser.add_argument("--max-tool-calls", type=int, default=gen["max_tool_calls"])
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--no-guard", action="store_true", help="disable the number-grounding guard")
    parser.add_argument("--no-constrain", action="store_true", help="disable constrained copy decoding")
    args = parser.parse_args()

    print(f"\n{'=' * 50}")
    print("  Milo ReAct Agent")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Tools: {', '.join(TOOLS.keys())}")
    print(f"{'=' * 50}")
    print("\nType your message (or 'quit' to exit, 'clear' to reset)\n")

    print("Loading model...", end=" ", flush=True)
    model = load_model(args.checkpoint, args.device)
    print("done!\n")

    conversation = []
    verbose = not args.no_verbose

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "clear":
            conversation = []
            print("(conversation cleared)\n")
            continue

        t0 = time.time()
        response, tool_calls = react_loop(
            model,
            user_input,
            conversation_history=conversation,
            device=args.device,
            max_tool_calls=args.max_tool_calls,
            temperature=args.temperature,
            verbose=verbose,
            guard=not args.no_guard,
            constrain=not args.no_constrain,
        )
        elapsed = time.time() - t0

        print(f"\nMilo: {response}")
        if tool_calls and verbose:
            print(f"  ({len(tool_calls)} tool call(s), {elapsed:.1f}s)")
        print()

        conversation.append(("user", user_input))
        conversation.append(("assistant", response))


if __name__ == "__main__":
    main()

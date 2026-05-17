THEME_1_AUTHORITY = {
    "id": "theme_1_authority",
    "title": "The Professor's Signature",

    "start": "scene_0",

    "scenes": {

        # =========================
        # SCENE 0 - INTRO
        # =========================
        "scene_0": {
            "id": "scene_0",
            "hati": "Hi, {user_name}. Today's scenario involves asking a strict professor for a signature. This is practice. Nothing here is real, but your feelings are valid. I'll be right here with you.",

            "options": [
                {"id": "begin", "label": "Begin", "next": "scene_1"}
            ]
        },

        # =========================
        # SCENE 1 - OFFICE + PIES
        # =========================
        "scene_1": {
            "id": "scene_1",

            "hati": (
                "You've entered the department office. The professor is busy at their desk. "
                "You need their signature on your data collection request form. You're waiting for your turn.\n\n"
                "Before we continue, let's do a P.I.E.S. check."
            ),

            "pies": True,

            "next": "scene_2"
        },

        # =========================
        # SCENE 2 - PREPARATION
        # =========================
        "scene_2": {
            "id": "scene_2",

            "hati": (
                "The other student is almost done. Let's prepare your line.\n\n"
                "You can say:\n"
                "A. Good morning, Professor. I need your signature on this form.\n"
                "B. Excuse me, could you sign this? It's for my thesis.\n"
                "C. Or type your own.\n\n"
                "Also prepare a short answer in case they ask about your research."
            ),

            "options": [
                {
                    "id": "script_a",
                    "label": "Good morning, Professor...",
                    "next": "scene_3"
                },
                {
                    "id": "script_b",
                    "label": "Excuse me, could you sign this?",
                    "next": "scene_3"
                },
                {
                    "id": "custom",
                    "label": "Custom response",
                    "next": "scene_3"
                }
            ]
        },

        # =========================
        # SCENE 3 - NPC INTERACTION
        # =========================
        "scene_3": {
            "id": "scene_3",

            "npc": {
                "name": "Professor",
                "line": "Yes? What is it?",
                "tone": "neutral_businesslike"
            },

            "branches": {

                # -------------------------
                # BRANCH A - CONFIDENT
                # -------------------------
                "calm": {
                    "hati": (
                        "Good. You were clear and direct.\n"
                        "That makes this much easier than your mind expected."
                    ),
                    "next": "scene_4_success"
                },

                # -------------------------
                # BRANCH B - ANXIOUS
                # -------------------------
                "anxious": {
                    "hati": (
                        "It's okay. Take a breath. Just say your line slowly."
                    ),
                    "next": "scene_4_mixed"
                },

                # -------------------------
                # BRANCH C - ANGER / DEFENSIVE
                # -------------------------
                "angry": {
                    "hati": (
                        "That reaction usually comes from stress. Let's reset.\n"
                        "A short apology can help de-escalate."
                    ),
                    "next": "scene_4_recovery"
                },

                # -------------------------
                # BRANCH D - FREEZE / AVOID
                # -------------------------
                "freeze": {
                    "hati": (
                        "You're freezing. That's a normal stress response.\n"
                        "Just one sentence is enough to continue."
                    ),
                    "next": "scene_4_avoid"
                }
            }
        },

        # =========================
        # SCENE 4 - OUTCOMES
        # =========================

        "scene_4_success": {
            "id": "scene_4_success",
            "hati": (
                "You got the signature.\n"
                "Notice how your body feels now compared to before."
            ),
            "next": "scene_5"
        },

        "scene_4_mixed": {
            "id": "scene_4_mixed",
            "hati": (
                "That felt hard, but you still completed the task.\n"
                "That's what matters."
            ),
            "next": "scene_5"
        },

        "scene_4_recovery": {
            "id": "scene_4_recovery",
            "hati": (
                "You recovered the interaction.\n"
                "The important part is that you corrected the moment."
            ),
            "next": "scene_5"
        },

        "scene_4_avoid": {
            "id": "scene_4_avoid",
            "hati": (
                "You stepped back. That's information, not failure.\n"
                "Next time, we can make the first step smaller."
            ),
            "next": "scene_5"
        },

        # =========================
        # SCENE 5 - DEBRIEF
        # =========================
        "scene_5": {
            "id": "scene_5",

            "hati": (
                "Let's reflect.\n\n"
                "Compare what you expected vs what actually happened.\n"
                "Did anything bad actually happen?\n"
                "What's one thing you did well today?"
            ),

            "inputs": [
                "expected_anxiety",
                "actual_anxiety",
                "reflection_text"
            ],

            "next": "scene_6"
        },

        # =========================
        # SCENE 6 - COPING STRATEGY
        # =========================
        "scene_6": {
            "id": "scene_6",

            "hati": (
                "Coping tool for next time:\n\n"
                "Fear → Grounding (feet on floor, slow breath)\n"
                "Anger → Label it: 'I'm stressed, not in danger'\n"
                "Freeze → Say just one word: 'Professor'\n\n"
                "Want to practice this next time?"
            ),

            "options": [
                {"id": "yes", "label": "Yes", "next": "scene_7"},
                {"id": "later", "label": "Maybe later", "next": "scene_7"}
            ]
        },

        # =========================
        # SCENE 7 - END
        # =========================
        "scene_7": {
            "id": "scene_7",

            "hati": (
                "You showed up today. That's what matters.\n"
                "Authority figures are just people doing their job.\n"
                "You handled a real stress simulation.\n\n"
                "I'll be here next time."
            ),

            "end": True
        }
    }
}
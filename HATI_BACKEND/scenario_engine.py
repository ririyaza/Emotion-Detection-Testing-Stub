import json
import os
import uuid


class ScenarioEngine:
    """
    Scripted scenario engine with branching and UI hints (buttons/text_input).
    Persists sessions to a JSON file for simple restarts.
    """

    def __init__(self, storage_path="scenario_sessions.json"):
        self.storage_path = storage_path
        self.sessions = {}
        self._load()

    def set_renderer(self, renderer):
        return None

    def start_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "step": "scene0_greet",
            "data": {}
        }
        self._save()
        return session_id, self._scene0_greet()

    def handle_step(self, session_id, payload):
        state = self.sessions.get(session_id)
        if not state:
            return {"error": "invalid session"}

        step = state["step"]
        data = state["data"]
        user_text = (payload.get("text") or "").strip() if payload else ""
        if payload and payload.get("emotion"):
            data["emotion"] = payload.get("emotion")

        if step == "scene0_greet":
            state["step"] = "scene0_intro"
            self._save()
            return self._scene0_intro()

        if step == "scene0_intro":
            state["step"] = "pies_physical"
            self._save()
            return self._scene1_intro()

        if step == "pies_physical":
            data["pies_physical"] = user_text
            state["step"] = "pies_emotional"
            self._save()
            return self._pies_emotional()

        if step == "pies_emotional":
            data["pies_emotional"] = user_text
            state["step"] = "pies_environmental"
            self._save()
            return self._pies_environmental()

        if step == "pies_environmental":
            data["pies_environmental"] = user_text
            state["step"] = "scene2_goal"
            self._save()
            return self._pies_response(data)

        if step == "scene2_goal":
            data["goal_text"] = user_text
            state["step"] = "scene2_line_choice"
            self._save()
            return self._scene2_line_choice()

        if step == "scene2_line_choice":
            data["line_choice"] = user_text
            if user_text.lower().startswith("custom"):
                state["step"] = "scene2_line_custom"
                self._save()
                return self._scene2_line_custom()
            state["step"] = "scene2_ready"
            self._save()
            return self._scene2_ready()

        if step == "scene2_line_custom":
            data["line_custom"] = user_text
            state["step"] = "scene2_ready"
            self._save()
            return self._scene2_ready()

        if step == "scene2_ready":
            state["step"] = "scene3_npc_prompt"
            self._save()
            return self._scene3_npc_prompt()

        if step == "scene3_npc_prompt":
            data["npc_choice"] = user_text
            if user_text.lower().startswith("custom"):
                state["step"] = "scene3_user_response"
                self._save()
                return self._scene3_user_response()
            data["npc_user_response"] = user_text
            branch = self._npc_branch(user_text, data.get("emotion", ""))
            data["npc_branch"] = branch
            state["step"] = "scene3_npc_reaction"
            self._save()
            return self._scene3_npc_reaction(branch)

        if step == "scene3_user_response":
            data["npc_user_response"] = user_text
            branch = self._npc_branch(user_text, data.get("emotion", ""))
            data["npc_branch"] = branch
            state["step"] = "scene3_npc_reaction"
            self._save()
            return self._scene3_npc_reaction(branch)

        if step == "scene3_npc_reaction":
            state["step"] = "scene4_debrief_intro"
            self._save()
            return self._scene4_debrief_intro()

        if step == "scene4_debrief_intro":
            state["step"] = "scene4_predicted"
            self._save()
            return self._scene4_predicted()

        if step == "scene4_predicted":
            data["predicted_anxiety"] = self._parse_number(user_text)
            state["step"] = "scene4_actual"
            self._save()
            return self._scene4_actual()

        if step == "scene4_actual":
            data["actual_anxiety"] = self._parse_number(user_text)
            state["step"] = "scene4_bad"
            self._save()
            return self._scene4_bad()

        if step == "scene4_bad":
            data["bad_happened"] = user_text
            if user_text.lower().startswith("y"):
                state["step"] = "scene4_bad_detail"
                self._save()
                return self._scene4_bad_detail()
            state["step"] = "scene4_credit"
            self._save()
            return self._scene4_credit()

        if step == "scene4_bad_detail":
            data["bad_detail"] = user_text
            state["step"] = "scene4_credit"
            self._save()
            return self._scene4_credit()

        if step == "scene4_credit":
            data["credit"] = user_text
            state["step"] = "scene4_personalized"
            self._save()
            return self._scene4_personalized(data)

        if step == "scene4_personalized":
            state["step"] = "scene4_reflection"
            self._save()
            return self._scene4_reflection()

        if step == "scene4_reflection":
            data["reflection"] = user_text
            state["step"] = "scene5_coping"
            self._save()
            return self._scene5_coping(data)

        if step == "scene5_coping":
            data["coping_try"] = user_text
            state["step"] = "scene6_closing"
            self._save()
            return self._scene6_closing(data)

        if step == "scene6_closing":
            state["step"] = "scene7_dashboard"
            self._save()
            return self._complete_and_dashboard()

        if step == "scene7_dashboard":
            state["step"] = "complete"
            self._save()
            return self._end()

        return {"error": "scenario complete"}


    def _scene0_greet(self):
        messages = [
            "Hi! Ready to practice? Remember, this is a safe space.",
            "Nothing here is real yet, but the feelings are valid.",
            "We will go through this together, one small step at a time.",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Begin"]})

    def _scene0_intro(self):
        messages = [
            "Today's scenario involves something many people find challenging—talking to someone new.",
            "I'll be here with you the whole time, helping you prepare and guiding you with coping strategies.",
            "You're not alone in this.",
        ]
        return self._payload(messages, {"type": "text_input", "placeholder": "Type anything to continue..."})

    def _scene1_intro(self):
        messages = [
            "You've just walked into your new class. It's the first day for your major subject, so the room is filled with mostly unfamiliar faces.",
            "The room is about half full. Some people are talking quietly, others are on their phones.",
            "There's an empty seat next to a student near the middle of the room. They haven't noticed you yet.",
            "This is your opportunity to practice approaching someone new.",
            "Before we do anything, let's check in. Physical: How does your body feel?",
        ]
        return self._payload(
            messages,
            {"type": "buttons", "options": ["Tense", "Relaxed", "Heart racing", "Shaky", "Normal"]},
        )

    def _pies_physical(self):
        return self._payload(
            ["Before we do anything, let's check in. Physical: How does your body feel?"],
            {"type": "buttons", "options": ["Tense", "Relaxed", "Heart racing", "Shaky", "Normal"]},
        )

    def _pies_emotional(self):
        return self._payload(
            ["Emotional: What are you feeling?"],
            {"type": "buttons", "options": ["Anxious", "Neutral", "Calm", "Irritable", "Sad"]},
        )

    def _pies_environmental(self):
        return self._payload(
            ["Environmental: What do you notice first?"],
            {"type": "buttons", "options": ["People staring", "The empty seat", "The exit door", "The stranger next to the seat"]},
        )

    def _pies_response(self, data):
        phys = (data.get("pies_physical") or "").lower()
        if phys in ["heart racing", "shaky", "tense"]:
            msg = (
                f"I notice you selected {data.get('pies_physical')}. "
                "That is completely normal when we are about to do something unfamiliar. "
                "Your body is just getting ready. Lets work with that, not against it."
            )
        else:
            msg = (
                "That is great that you are feeling calm right now. "
                "That is a resource we can use. Lets see if we can keep that feeling as we go through this."
            )
        messages = [
            msg,
            "Before you approach that empty seat, think about what you want to happen.",
            "You do not have to become best friends with this person. The goal can be smaller.",
            "What would feel like a small win for you right now?",
        ]
        return self._payload(messages, {"type": "text_input", "placeholder": "Type your goal..."})

    def _scene2_goal(self):
        messages = [
            "Before you approach that empty seat, think about what you want to happen.",
            "You do not have to become best friends with this person. The goal can be smaller.",
            "What would feel like a small win for you right now?",
        ]
        return self._payload(messages, {"type": "text_input", "placeholder": "Type your goal..."})

    def _scene2_line_choice(self):
        messages = [
            "That's a good goal. Keep that in mind.",
            "Now, when you walk toward that seat, the person next to it will probably notice you. They might look up. That's normal.",
            "Let's think about what you could say—just a simple line, nothing fancy.",
            "Here are some options, or you can create your own:",
        ]
        return self._payload(
            messages,
            {"type": "buttons", "options": [
                "Is this seat taken?",
                "Hey, is anyone sitting here?",
                "Just a nod and smile, no words",
                "Custom response"
            ]},
        )

    def _scene2_line_custom(self):
        return self._payload(["Type your custom line:"], {"type": "text_input", "placeholder": "Your line..."})

    def _scene2_ready(self):
        messages = [
            "Good. You have your line. Now let's take a breath together before you go.",
            "Remember: you're not performing. You're just a person existing in a space, like everyone else.",
            "Ready?",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Approach Seat"]})

    def _scene3_npc_prompt(self):
        messages = [
            "The student looks up.",
            "NPC: Oh, hey. Did you need this seat?",
            "How do you respond?",
        ]
        return self._payload(
            messages,
            {"type": "buttons", "options": [
                "Yeah, is it free?",
                "Thanks, yeah. I'm new here, by the way.",
                "Just nod and smile while sitting",
                "Custom response"
            ]},
        )

    def _scene3_user_response(self):
        return self._payload(["Type your response:"], {"type": "text_input", "placeholder": "Your response..."})

    def _scene3_npc_reaction(self, branch):
        if branch == "risk":
            messages = [
                "NPC: Yeah, go for it. I'm new here too. This class seems intense, huh?",
                "Hati: Nice. They responded warmly.",
                "Notice how that feels in your body right now compared to before. This is good practice—you just did it.",
            ]
        elif branch == "nod":
            messages = [
                "NPC nods back and returns to their phone.",
                "Hati: That is okay. You did the minimum and it was fine. No rejection, no drama.",
                "You are in the seat. That is a win. If you want, you could say something later—but no pressure.",
            ]
        else:
            messages = [
                "NPC: \"Oh... okay. No problem.\"",
                "Hati: That felt awkward, I know. But look—nothing bad happened. They were not mean.",
                "You are still here, and we can try again. Let's take a breath. What do you need right now?",
            ]
        return self._payload(messages, {"type": "text_input", "placeholder": "Type anything to continue..."})

    def _scene4_debrief_intro(self):
        messages = [
            "Okay. Let's pause and reflect. You just did something that takes courage—you approached a stranger.",
            "Let's compare what you predicted with what actually happened.",
        ]
        return self._payload(messages, {"type": "text_input", "placeholder": "Type anything to continue..."})

    def _scene4_predicted(self):
        return self._payload(
            ["On a scale of 0-10, how anxious did you expect to feel during that interaction?"],
            {"type": "text_input", "placeholder": "Type a number 0-10..."},
        )

    def _scene4_actual(self):
        return self._payload(
            ["On a scale of 0-10, how anxious did you actually feel?"],
            {"type": "text_input", "placeholder": "Type a number 0-10..."},
        )

    def _scene4_bad(self):
        return self._payload(
            ["Did anything bad actually happen?"],
            {"type": "buttons", "options": ["Yes", "No", "Not sure"]},
        )

    def _scene4_bad_detail(self):
        return self._payload(["What happened?"], {"type": "text_input", "placeholder": "Type your response..."})

    def _scene4_credit(self):
        return self._payload(
            ["What's one small thing you can give yourself credit for?"],
            {"type": "text_input", "placeholder": "Type your response..."},
        )

    def _scene4_personalized(self, data):
        pred = data.get("predicted_anxiety")
        actual = data.get("actual_anxiety")
        if pred is None or actual is None:
            msg = "Thank you for being honest. You showed up and practiced."
        elif actual < pred:
            msg = (
                "Look at that—your actual anxiety was lower than you predicted. That's evidence. "
                "Your brain predicted danger, but reality was safer or more neutral. "
                "Every time this happens, you teach your brain a new lesson: you can handle this."
            )
        elif actual == pred:
            msg = (
                "Thank you for being honest. It is still hard, and that is okay. "
                "What matters is that you did it anyway. That is courage—feeling the fear and doing it."
            )
        else:
            msg = (
                "I appreciate your honesty. Today was harder than expected, and that happens. "
                "The fact that you stayed and tried anyway is resilience."
            )
        return self._payload([msg], {"type": "text_input", "placeholder": "Type anything to continue..."})

    def _scene4_reflection(self):
        messages = [
            "Earlier, you were worried about how the other person would respond.",
            "Now that it's over, what do you notice about them? Were they scary, friendly, neutral?",
            "Most people are focused on themselves, not judging you.",
        ]
        return self._payload(messages, {"type": "text_input", "placeholder": "Type your response..."})

    def _scene5_coping(self, data):
        emotion = (data.get("emotion") or "").lower()
        if emotion in ["anxious", "fear"]:
            tool = "Box breathing: breathe in 4, hold 4, out 4, hold 4. Try it once now."
        elif emotion in ["anger", "angry", "disgust"]:
            tool = "5-4-3-2-1 grounding: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste."
        elif emotion in ["sad"]:
            tool = "Opposite action: even if you do not feel like it, give one small smile to the person next to you."
        elif emotion in ["neutral"]:
            tool = "Body scan: notice your feet on the floor, your back against the chair. You are here."
        else:
            tool = "Savoring: stay with this good feeling for 10 seconds. Let it sink in."
        messages = [
            "Based on what you just experienced, here is a small tool to carry with you:",
            tool,
            "Want to try this together now?",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Yes", "Maybe later"]})

    def _scene6_closing(self, data):
        messages = [
            "Here's what I want you to remember from today:",
            "You showed up. You practiced. You're building skills, one moment at a time. I'm proud of you.",
            "For now, take a deep breath. You've earned a moment of rest.",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Finish"]})

    def _scene7_dashboard(self):
        messages = [
            "Great work today. I've logged your emotions and progress.",
            "Over time, you'll start to see patterns, what triggers your anxiety, what helps, and how far you've come.",
            "Want to see your progress so far?",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Open Progress", "Close"]})

    def _complete(self):
        return self._payload(
            ["CONGRATULATIONS! YOU JUST COMPLETED A SCENARIO!"],
            {"type": "text_input"},
        )

    def _complete_and_dashboard(self):
        messages = [
            "CONGRATULATIONS! YOU JUST COMPLETED A SCENARIO!",
            "Great work today. I've logged your emotions and progress.",
            "Over time, you'll start to see patternsâ€”what triggers your anxiety, what helps, and how far you've come.",
            "Want to see your progress so far?",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Open Progress", "Close"]})

    def _end(self):
        return self._payload([], {"type": "text_input"})


    def _npc_branch(self, response_text, emotion):
        text = (response_text or "").lower()
        if "nod" in text or "smile" in text:
            return "nod"
        if any(k in text for k in ["sorry", "never mind", "nevermind", "uh", "um"]):
            return "high_anxiety"
        if emotion in ["sad", "anxious", "fear"]:
            return "high_anxiety"
        return "risk"

    def _parse_number(self, text):
        if not text:
            return None
        for token in text.replace("/", " ").replace(":", " ").split():
            try:
                val = int(token)
                if 0 <= val <= 10:
                    return val
            except Exception:
                continue
        return None

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.sessions = data
        except Exception:
            self.sessions = {}

    def _save(self):
        tmp_path = self.storage_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.sessions, f)
            os.replace(tmp_path, self.storage_path)
        except Exception:
            pass

    def _payload(self, messages, ui):
        msgs = [m for m in (messages or []) if m]
        first = msgs[0] if msgs else ""
        return {"messages": msgs, "message": first, "ui": ui}


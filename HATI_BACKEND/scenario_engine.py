import json
import os
import uuid

THEME_SCENARIO_KEYS = {
    "Fear of Authority": "foa_supervisor",
    "Fear of Negative Evaluation & Embarrassment": "fne_stage",
    "Physiological Symptoms": "phys_classroom",
    "Fear of Social Gatherings": "fsg_party",
    "Fear of Strangers & New People": "fsn_seat",
    "Fear of Being Observed & Performing": "fbop_spotlight",
}


ALLOWED_SCENARIO_KEYS = {
    "Fear of Authority": {"foa_supervisor", "foa_classroom"},
    "Fear of Negative Evaluation & Embarrassment": {"fne_stage"},
    "Physiological Symptoms": {"phys_classroom"},
    "Fear of Social Gatherings": {"fsg_party"},
    "Fear of Strangers & New People": {"fsn_seat", "fsn_classroom"},
    "Fear of Being Observed & Performing": {"fbop_spotlight"},
}


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

    def start_session(self, theme="", scenario_key="", user_name=""):
        theme_key = self._normalize_theme(theme)
        sk = self._resolve_scenario_key(theme_key, scenario_key or "")
        session_id = str(uuid.uuid4())
        name = (user_name or "").strip() or "there"
        self.sessions[session_id] = {
            "step": "scene0_greet",
            "data": {"theme": theme_key, "scenario_key": sk, "user_name": name},
        }
        self._save()
        return session_id, self._scene0_pre_scenario(theme_key, sk, name)

    def handle_step(self, session_id, payload):
        state = self.sessions.get(session_id)
        if not state:
            return {"error": "invalid session"}

        step = state["step"]
        data = state["data"]
        user_text = (payload.get("text") or "").strip() if payload else ""
        if payload:
            incoming_name = (payload.get("user_name") or payload.get("display_name") or "").strip()
            if incoming_name:
                data["user_name"] = incoming_name
        if payload and "emotion" in payload:
            data["emotion"] = (payload.get("emotion") or "").strip()

        if step == "scene0_greet":
            state["step"] = "pies_physical"
            self._save()
            return self._scene1_intro(
                data.get("theme", ""),
                data.get("scenario_key", ""),
            )

        if step == "pies_physical":
            data["pies_physical"] = user_text
            state["step"] = "pies_emotional"
            self._save()
            return self._pies_emotional(data)

        if step == "pies_emotional":
            data["pies_emotional"] = user_text
            state["step"] = "pies_environmental"
            self._save()
            return self._pies_environmental(data)

        if step == "pies_environmental":
            data["pies_environmental"] = user_text
            nxt, out = self._after_pies_environmental(data)
            state["step"] = nxt
            self._save()
            return out

        # --- Fear of Authority: Professor's Signature ---
        if step == "foa_s2_script":
            data["foa_script_pick"] = user_text
            ut = user_text.lower()
            if ut.startswith("c:") or "i'll type" in ut or "type my own" in ut or ut.startswith("custom"):
                state["step"] = "foa_s2_script_custom"
                self._save()
                return self._payload(
                    ["**Hati:** Type the line you'll say to the professor:"],
                    {"type": "text_input", "placeholder": "Your line..."},
                )
            state["step"] = "foa_s2_q_prep"
            self._save()
            return self._foa_s2_q_prep_view()

        if step == "foa_s2_script_custom":
            data["foa_script_custom"] = user_text
            state["step"] = "foa_s2_q_prep"
            self._save()
            return self._foa_s2_q_prep_view()

        if step == "foa_s2_q_prep":
            data["foa_q_prep"] = user_text
            state["step"] = "foa_s2_ready"
            self._save()
            return self._foa_s2_ready_view()

        if step == "foa_s2_ready":
            state["step"] = "foa_s3_npc"
            self._save()
            return self._foa_s3_npc_view()

        if step == "foa_s3_npc":
            data["foa_npc_pick"] = user_text
            data["npc_user_response"] = user_text
            data["story_branch"] = self._foa_story_branch(user_text, data.get("emotion", ""))
            data["foa_r_phase"] = 0
            state["step"] = "foa_s3_reaction"
            self._save()
            return self._foa_s3_reaction_enter(data)

        if step == "foa_s3_custom":
            data["npc_user_response"] = user_text
            data["story_branch"] = self._foa_story_branch(user_text, data.get("emotion", ""))
            data["foa_r_phase"] = 0
            state["step"] = "foa_s3_reaction"
            self._save()
            return self._foa_s3_reaction_enter(data)

        if step == "foa_s3_reaction":
            out = self._advance_foa_s3_reaction(state, data, user_text)
            self._save()
            return out

        # --- Fear of Strangers: Food Hall ---
        if step == "fsn_s2_practice":
            state["step"] = "fsn_s2_goal"
            self._save()
            return self._fsn_s2_goal_prompt_view()

        if step == "fsn_s2_goal":
            data["goal_text"] = user_text
            state["step"] = "fsn_s2_ready"
            self._save()
            return self._fsn_s2_ready_view()

        if step == "fsn_s2_ready":
            state["step"] = "fsn_s3_npc"
            self._save()
            return self._fsn_s3_npc_view()

        if step == "fsn_s3_npc":
            data["fsn_npc_pick"] = user_text
            data["npc_user_response"] = user_text
            data["story_branch"] = self._fsn_story_branch(user_text, data.get("emotion", ""))
            data["fsn_r_phase"] = 0
            state["step"] = "fsn_s3_reaction"
            self._save()
            return self._fsn_s3_reaction_enter(data)

        if step == "fsn_s3_custom":
            data["npc_user_response"] = user_text
            data["story_branch"] = self._fsn_story_branch(user_text, data.get("emotion", ""))
            data["fsn_r_phase"] = 0
            state["step"] = "fsn_s3_reaction"
            self._save()
            return self._fsn_s3_reaction_enter(data)

        if step == "fsn_s3_reaction":
            out = self._advance_fsn_s3_reaction(state, data, user_text)
            self._save()
            return out

        # --- Thesis defense (observed / performing) ---
        if step == "fbop_s2_title":
            data["fbop_title"] = user_text
            state["step"] = "fbop_s2_opening"
            self._save()
            return self._fbop_s2_opening_view(data)

        if step == "fbop_s2_opening":
            data["fbop_opening_pick"] = user_text
            if "custom" in user_text.lower():
                state["step"] = "fbop_s2_opening_custom"
                self._save()
                return self._payload(
                    ["Type your opening line:"],
                    {"type": "text_input", "placeholder": "Opening..."},
                )
            state["step"] = "fbop_s2_pause"
            self._save()
            return self._fbop_s2_pause_view()

        if step == "fbop_s2_opening_custom":
            data["fbop_opening_custom"] = user_text
            state["step"] = "fbop_s2_pause"
            self._save()
            return self._fbop_s2_pause_view()

        if step == "fbop_s2_pause":
            state["step"] = "fbop_s2_contrib"
            self._save()
            return self._fbop_s2_contrib_view()

        if step == "fbop_s2_contrib":
            data["fbop_contrib"] = user_text
            state["step"] = "fbop_s2_goal"
            self._save()
            return self._fbop_s2_goal_view()

        if step == "fbop_s2_goal":
            data["fbop_goal_pick"] = user_text
            if "custom" in user_text.lower():
                state["step"] = "fbop_s2_goal_custom"
                self._save()
                return self._payload(
                    ["Type your small win goal:"],
                    {"type": "text_input", "placeholder": "Goal..."},
                )
            data["goal_text"] = user_text
            state["step"] = "fbop_s2_ground"
            self._save()
            return self._fbop_s2_ground_view()

        if step == "fbop_s2_goal_custom":
            data["goal_text"] = user_text
            state["step"] = "fbop_s2_ground"
            self._save()
            return self._fbop_s2_ground_view()

        if step == "fbop_s2_ground":
            data["fbop_ground"] = user_text
            state["step"] = "fbop_s2_ready"
            self._save()
            return self._fbop_s2_ready_view()

        if step == "fbop_s2_ready":
            state["step"] = "fbop_s3_delivery"
            self._save()
            return self._fbop_s3_delivery_view()

        if step == "fbop_s3_delivery":
            data["fbop_delivery"] = user_text
            data["story_branch"] = self._fbop_story_branch(user_text, data.get("emotion", ""))
            data["fbop_o_phase"] = 0
            state["step"] = "fbop_s3_outcome"
            self._save()
            return self._fbop_s3_outcome_enter(data)

        if step == "fbop_s3_outcome":
            out = self._advance_fbop_s3_outcome(state, data, user_text)
            self._save()
            return out

        # --- House party (social gatherings) ---
        if step == "fsg_s2_path":
            data["fsg_path"] = user_text
            if "path b" in user_text.lower() or "corner" in user_text.lower() or "b)" in user_text.lower():
                state["step"] = "fsg_b_goal"
                self._save()
                return self._fsg_b_goal_view()
            state["step"] = "fsg_a_opening"
            self._save()
            return self._fsg_a_opening_view(data)

        if step == "fsg_a_opening":
            data["fsg_open_pick"] = user_text
            if "custom" in user_text.lower():
                state["step"] = "fsg_a_opening_custom"
                self._save()
                return self._payload(
                    ["Type your opening line:"],
                    {"type": "text_input", "placeholder": "Opening..."},
                )
            state["step"] = "fsg_a_practice_pause"
            self._save()
            return self._fsg_a_practice_pause_view()

        if step == "fsg_a_opening_custom":
            data["fsg_open_custom"] = user_text
            state["step"] = "fsg_a_practice_pause"
            self._save()
            return self._fsg_a_practice_pause_view()

        if step == "fsg_a_practice_pause":
            state["step"] = "fsg_a_relation"
            self._save()
            return self._fsg_a_relation_view()

        if step == "fsg_a_relation":
            data["fsg_relation"] = user_text
            state["step"] = "fsg_a_goal"
            self._save()
            return self._fsg_a_goal_view()

        if step == "fsg_a_goal":
            data["fsg_goal_pick"] = user_text
            if "custom" in user_text.lower():
                state["step"] = "fsg_a_goal_custom"
                self._save()
                return self._payload(
                    ["Type your small win goal:"],
                    {"type": "text_input", "placeholder": "Goal..."},
                )
            data["goal_text"] = user_text
            state["step"] = "fsg_ground"
            self._save()
            return self._fsg_ground_view()

        if step == "fsg_a_goal_custom":
            data["goal_text"] = user_text
            state["step"] = "fsg_ground"
            self._save()
            return self._fsg_ground_view()

        if step == "fsg_b_goal":
            data["fsg_b_goal_pick"] = user_text
            if "custom" in user_text.lower():
                state["step"] = "fsg_b_goal_custom"
                self._save()
                return self._payload(
                    ["Type your small win goal:"],
                    {"type": "text_input", "placeholder": "Goal..."},
                )
            data["goal_text"] = user_text
            state["step"] = "fsg_ground"
            self._save()
            return self._fsg_ground_view()

        if step == "fsg_b_goal_custom":
            data["goal_text"] = user_text
            state["step"] = "fsg_ground"
            self._save()
            return self._fsg_ground_view()

        if step == "fsg_ground":
            data["fsg_ground"] = user_text
            state["step"] = "fsg_s2_proceed"
            self._save()
            return self._fsg_s2_proceed_view()

        if step == "fsg_s2_proceed":
            state["step"] = "fsg_s3_social"
            self._save()
            return self._fsg_s3_social_view(data)

        if step == "fsg_s3_social":
            path = (data.get("fsg_path") or "").lower()
            is_corner = "path b" in path or "corner" in path
            if is_corner:
                data["fsg_social_pick"] = user_text
                sp = (user_text or "").lower()
                if "approach" in sp and "group" in sp:
                    data["fsg_opening_flow"] = True
                else:
                    data.pop("fsg_opening_flow", None)
                data["story_branch"] = self._fsg_story_branch(
                    user_text, data.get("emotion", ""), data.get("fsg_path", "")
                )
            else:
                data["fsg_delivered_opening"] = user_text
                data["story_branch"] = self._fsg_branch_from_opening_delivery(
                    user_text,
                    data.get("emotion", ""),
                    data.get("pies_emotional", ""),
                )
                data.pop("fsg_opening_flow", None)
            data["fsg_rx_ph"] = 0
            state["step"] = "fsg_s3_reaction"
            self._save()
            return self._advance_fsg_s3_reaction(state, data, None)

        if step == "fsg_s3_reaction":
            out = self._advance_fsg_s3_reaction(state, data, user_text)
            self._save()
            return out

        # --- Group project (negative evaluation) ---
        if step == "fne_s2_style":
            data["fne_style_pick"] = user_text
            if "custom" in user_text.lower():
                state["step"] = "fne_s2_style_custom"
                self._save()
                return self._payload(
                    ["Type how you'll open your part:"],
                    {"type": "text_input", "placeholder": "Your opening..."},
                )
            state["step"] = "fne_s2_pause"
            self._save()
            return self._fne_s2_pause_view()

        if step == "fne_s2_style_custom":
            data["fne_style_custom"] = user_text
            state["step"] = "fne_s2_pause"
            self._save()
            return self._fne_s2_pause_view()

        if step == "fne_s2_pause":
            state["step"] = "fne_s2_points"
            self._save()
            return self._fne_s2_points_view()

        if step == "fne_s2_points":
            data["fne_points"] = user_text
            state["step"] = "fne_s2_goal"
            self._save()
            return self._fne_s2_goal_view()

        if step == "fne_s2_goal":
            data["fne_goal_pick"] = user_text
            if "custom" in user_text.lower():
                state["step"] = "fne_s2_goal_custom"
                self._save()
                return self._payload(
                    ["Type your small win goal:"],
                    {"type": "text_input", "placeholder": "Goal..."},
                )
            data["goal_text"] = user_text
            state["step"] = "fne_s2_ground"
            self._save()
            return self._fne_s2_ground_view()

        if step == "fne_s2_goal_custom":
            data["goal_text"] = user_text
            state["step"] = "fne_s2_ground"
            self._save()
            return self._fne_s2_ground_view()

        if step == "fne_s2_ground":
            data["fne_ground"] = user_text
            state["step"] = "fne_s2_ready"
            self._save()
            return self._fne_s2_ready_view()

        if step == "fne_s2_ready":
            state["step"] = "fne_s3_carlo"
            self._save()
            return self._fne_s3_carlo_view()

        if step == "fne_s3_carlo":
            data["fne_carlo_response"] = user_text
            data["story_branch"] = self._fne_story_branch(user_text, data.get("emotion", ""))
            data["fne_outcome_phase"] = 0
            state["step"] = "fne_s3_outcome"
            self._save()
            return self._fne_s3_outcome_view(data)

        if step == "fne_s3_outcome":
            br = data.get("story_branch", "")
            ph = int(data.get("fne_outcome_phase", 0))
            if br == "apologetic":
                if ph == 0:
                    data["fne_outcome_phase"] = 1
                    ut = (user_text or "").lower()
                    if "friday" in ut or "move on" in ut:
                        msgs = [
                            "Carlo: Alright.",
                            "Precious: I think we can work with that timeline.",
                            "Julia: Let's keep going.",
                        ]
                    else:
                        msgs = [
                            "Carlo: Okay.",
                            "Precious: Let's change the subject for a minute.",
                        ]
                    return self._payload(
                        msgs
                        + [
                            "**Hati:** Notice how different responses land. Apology without a boundary can invite more pressure; a calm plan steadies the room.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                state["step"] = "scene4_debrief_intro"
                self._save()
                return self._scene4_debrief_intro(data)
            state["step"] = "scene4_debrief_intro"
            self._save()
            return self._scene4_debrief_intro(data)

        # --- Physiological / bus stop ---
        if step == "phys_s2_excuse":
            data["phys_excuse_pick"] = user_text
            if "custom" in user_text.lower():
                state["step"] = "phys_s2_excuse_custom"
                self._save()
                return self._payload(
                    ["Type your line:"],
                    {"type": "text_input", "placeholder": "Your line..."},
                )
            state["step"] = "phys_s2_ack"
            self._save()
            return self._phys_s2_ack_view()

        if step == "phys_s2_excuse_custom":
            data["phys_excuse_custom"] = user_text
            state["step"] = "phys_s2_ack"
            self._save()
            return self._phys_s2_ack_view()

        if step == "phys_s2_ack":
            state["step"] = "phys_s2_goal"
            self._save()
            return self._phys_s2_goal_view()

        if step == "phys_s2_goal":
            data["phys_goal_pick"] = user_text
            if "custom" in user_text.lower():
                state["step"] = "phys_s2_goal_custom"
                self._save()
                return self._payload(
                    ["Type your small win goal:"],
                    {"type": "text_input", "placeholder": "Goal..."},
                )
            data["goal_text"] = user_text
            state["step"] = "phys_s2_ground"
            self._save()
            return self._phys_s2_ground_view()

        if step == "phys_s2_goal_custom":
            data["goal_text"] = user_text
            state["step"] = "phys_s2_ground"
            self._save()
            return self._phys_s2_ground_view()

        if step == "phys_s2_ground":
            data["phys_ground"] = user_text
            state["step"] = "phys_s2_ready"
            self._save()
            return self._phys_s2_ready_view()

        if step == "phys_s2_ready":
            state["step"] = "phys_s3_classmate"
            self._save()
            return self._phys_s3_classmate_view()

        if step == "phys_s3_classmate":
            data["phys_reply_pick"] = user_text
            if "custom" in user_text.lower():
                state["step"] = "phys_s3_custom"
                self._save()
                return self._payload(
                    ["Type your reply:"],
                    {"type": "text_input", "placeholder": "Reply..."},
                )
            data["npc_user_response"] = user_text
            data["story_branch"] = self._phys_story_branch(user_text, data.get("emotion", ""))
            data["phys_r_ph"] = 0
            state["step"] = "phys_s3_reaction"
            self._save()
            return self._advance_phys_s3_reaction(state, data, None)

        if step == "phys_s3_custom":
            data["npc_user_response"] = user_text
            data["story_branch"] = self._phys_story_branch(user_text, data.get("emotion", ""))
            data["phys_r_ph"] = 0
            state["step"] = "phys_s3_reaction"
            self._save()
            return self._advance_phys_s3_reaction(state, data, None)

        if step == "phys_s3_reaction":
            out = self._advance_phys_s3_reaction(state, data, user_text)
            self._save()
            return out

        # --- WHERE TO SIT / generic seat flow (foa_classroom) ---
        if step == "scene2_goal":
            data["goal_text"] = user_text
            state["step"] = "scene2_line_choice"
            self._save()
            return self._scene2_line_choice(data)

        if step == "scene2_line_choice":
            data["line_choice"] = user_text
            ut = user_text.lower()
            if ut.startswith("custom") or "type my own" in ut or "i'll type" in ut:
                state["step"] = "scene2_line_custom"
                self._save()
                return self._scene2_line_custom()
            state["step"] = "scene2_ready"
            self._save()
            return self._scene2_ready(data)

        if step == "scene2_line_custom":
            data["line_custom"] = user_text
            state["step"] = "scene2_ready"
            self._save()
            return self._scene2_ready(data)

        if step == "scene2_ready":
            state["step"] = "scene3_npc_prompt"
            self._save()
            return self._scene3_npc_prompt(data)

        if step == "scene3_npc_prompt":
            data["npc_choice"] = user_text
            data["npc_user_response"] = user_text
            branch = self._npc_branch_seat(user_text, data.get("emotion", ""))
            data["npc_branch"] = branch
            data["story_branch"] = "anxious" if branch == "high_anxiety" else ("confident" if branch == "risk" else "freeze")
            state["step"] = "scene3_npc_reaction"
            self._save()
            return self._scene3_npc_reaction(data, branch)

        if step == "scene3_user_response":
            data["npc_user_response"] = user_text
            branch = self._npc_branch_seat(user_text, data.get("emotion", ""))
            data["npc_branch"] = branch
            data["story_branch"] = "anxious" if branch == "high_anxiety" else ("confident" if branch == "risk" else "freeze")
            state["step"] = "scene3_npc_reaction"
            self._save()
            return self._scene3_npc_reaction(data, branch)

        if step == "scene3_npc_reaction":
            sk = data.get("scenario_key", "")
            br = data.get("npc_branch")
            if sk == "fsn_classroom" and br == "high_anxiety":
                ut = user_text.lower()
                if "try again" in ut:
                    state["step"] = "scene3_npc_prompt"
                    self._save()
                    return self._scene3_npc_prompt(data)
                state["step"] = "scene4_debrief_intro"
                self._save()
                return self._scene4_debrief_intro(data)
            state["step"] = "scene4_debrief_intro"
            self._save()
            return self._scene4_debrief_intro(data)

        if step == "scene4_debrief_intro":
            state["step"] = "scene4_predicted"
            self._save()
            return self._scene4_predicted(data)

        if step == "scene4_predicted":
            data["predicted_anxiety"] = self._parse_number(user_text)
            state["step"] = "scene4_actual"
            self._save()
            return self._scene4_actual(data)

        if step == "scene4_actual":
            data["actual_anxiety"] = self._parse_number(user_text)
            theme = data.get("theme", "")
            if theme == "Fear of Social Gatherings":
                state["step"] = "scene4_fsg_cause"
                self._save()
                return self._scene4_fsg_cause()
            if theme == "Fear of Negative Evaluation & Embarrassment":
                state["step"] = "scene4_fne_observe"
                self._save()
                return self._scene4_fne_observe()
            state["step"] = "scene4_bad"
            self._save()
            return self._scene4_bad()

        if step == "scene4_fsg_cause":
            data["fsg_cause"] = user_text
            state["step"] = "scene4_bad"
            self._save()
            return self._scene4_bad()

        if step == "scene4_fne_observe":
            data["fne_outcome_observe"] = user_text
            state["step"] = "scene4_fne_severity"
            self._save()
            return self._scene4_fne_severity()

        if step == "scene4_fne_severity":
            data["fne_outcome_severity"] = self._parse_number(user_text)
            state["step"] = "scene4_fne_goal_done"
            self._save()
            return self._scene4_fne_goal_done_view(data)

        if step == "scene4_fne_goal_done":
            data["fne_goal_achieved"] = user_text
            state["step"] = "scene4_credit"
            self._save()
            return self._scene4_credit()

        if step == "scene4_bad":
            data["bad_happened"] = user_text
            if user_text.lower().startswith("y"):
                state["step"] = "scene4_bad_detail"
                self._save()
                return self._scene4_bad_detail()
            if data.get("theme") == "Fear of Social Gatherings":
                state["step"] = "scene4_fsg_goal_done"
                self._save()
                return self._scene4_fsg_goal_done_view(data)
            state["step"] = "scene4_credit"
            self._save()
            return self._scene4_credit()

        if step == "scene4_bad_detail":
            data["bad_detail"] = user_text
            if data.get("theme") == "Fear of Social Gatherings":
                state["step"] = "scene4_fsg_badness"
                self._save()
                return self._scene4_fsg_badness()
            state["step"] = "scene4_credit"
            self._save()
            return self._scene4_credit()

        if step == "scene4_fsg_badness":
            data["fsg_badness"] = self._parse_number(user_text)
            state["step"] = "scene4_fsg_goal_done"
            self._save()
            return self._scene4_fsg_goal_done_view(data)

        if step == "scene4_fsg_goal_done":
            data["fsg_goal_achieved"] = user_text
            state["step"] = "scene4_credit"
            self._save()
            return self._scene4_credit()

        if step == "scene4_credit":
            data["credit"] = user_text
            state["step"] = "scene4_personalized"
            self._save()
            return self._scene4_personalized(data)

        if step == "scene4_personalized":
            if data.get("scenario_key") == "fsn_classroom":
                state["step"] = "scene4_class_npc_reflect"
                self._save()
                return self._scene4_class_npc_reflect_view()
            state["step"] = "scene4_reflection"
            self._save()
            return self._scene4_reflection(data)

        if step == "scene4_class_npc_reflect":
            lt = (user_text or "").strip().lower()
            if lt and lt != "skip":
                data["class_npc_reflection"] = user_text
            state["step"] = "scene4_class_npc_done"
            self._save()
            return self._payload(
                [
                    "**Hati:** That's a really important observation. Keep noticing that pattern—your predictions are often worse than reality.",
                ],
                {"type": "buttons", "options": ["Continue"]},
            )

        if step == "scene4_class_npc_done":
            state["step"] = "scene5_coping"
            self._save()
            return self._scene5_coping(data)

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
            return self._complete_and_dashboard(data)

        if step == "scene7_dashboard":
            state["step"] = "complete"
            self._save()
            return self._end()

        return {"error": "scenario complete"}


    def _scene0_pre_scenario(self, theme, scenario_key, name="there"):
        """SCENE 0 verbatim lines + Begin (no text input before Begin)."""
        n = name or "there"
        if theme == "Fear of Strangers & New People" and scenario_key == "fsn_classroom":
            messages = [
                f"**Hati:** Hi, {n}! Ready to practice? Remember, this is a safe space. Nothing here is real… yet, but the feelings are valid. We'll go through this together, one small step at a time.",
                "**Hati:** Today's scenario involves something many people find challenging—talking to someone new. I'll be here with you the whole time, helping you prepare and guiding you with coping strategies. You're not alone in this.",
            ]
            return self._payload(messages, {"type": "buttons", "options": ["Begin"]})
        if theme == "Fear of Authority" and scenario_key == "foa_classroom":
            messages = [
                f"**Hati:** Hi, {n}. Today's scenario is: WHERE TO SIT?—practice choosing a seat near someone new on the first day of class.",
                "**Hati:** Remember, this is practice. Nothing here is real, but the feelings are valid. I'll be right here with you.",
            ]
            return self._payload(messages, {"type": "buttons", "options": ["Begin"]})
        if theme == "Fear of Authority":
            messages = [
                f"**Hati:** Hi, {n}. Today's scenario involves something many students find challenging—asking a strict professor for a signature.",
                "**Hati:** Remember, this is practice. Nothing here is real, but the feelings are valid. I'll be right here with you.",
            ]
            return self._payload(messages, {"type": "buttons", "options": ["Begin"]})
        if theme == "Fear of Strangers & New People":
            messages = [
                f"**Hati:** Hey, {n}. Today's scenario is about approaching a stranger in a public place, a food hall where you'll be eating! This is a common trigger for social anxiety, but it's also a low-risk way to practice. I'll be right here with you. Ready?",
            ]
            return self._payload(messages, {"type": "buttons", "options": ["Begin"]})
        if theme == "Fear of Being Observed & Performing":
            messages = [
                f"**Hati:** Hi, {n}. Today's scenario is a big one: presenting your research objectives to a thesis panel. This is a classic performance situation. The fear is real, but you can practice handling it. I'll be right here with you.",
            ]
            return self._payload(messages, {"type": "buttons", "options": ["Begin"]})
        if theme == "Fear of Social Gatherings":
            messages = [
                f"**Hati:** Hey, {n}. Today's scenario is a party – a friend's birthday. You're arriving alone, and you don't know most of the guests. This is a classic social gathering trigger. I'll be right here with you step by step.",
            ]
            return self._payload(messages, {"type": "buttons", "options": ["Begin"]})
        if theme == "Fear of Negative Evaluation & Embarrassment":
            messages = [
                f"**Hati:** Hey, {n}. Today's scenario is a group project meeting. You're sharing your part of the presentation, but you're not fully ready. After you speak, someone criticizes you. This is a classic trigger for fear of negative evaluation. I'll be right here with you.",
            ]
            return self._payload(messages, {"type": "buttons", "options": ["Begin"]})
        if theme == "Physiological Symptoms":
            messages = [
                f"**Hati:** Hey, {n}. Today's scenario is about physical anxiety symptoms – sweating, heart racing, shaking – happening in public. An old classmate notices and asks if you're okay. You don't want to reveal your anxiety. Let's practice handling this. I'm right here.",
            ]
            return self._payload(messages, {"type": "buttons", "options": ["Begin"]})
        messages = [
            f"**Hati:** Hi, {n}. I'll walk you through a short practice scenario.",
            "**Hati:** Nothing here is real, but the feelings are valid. I'll be right here with you.",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Begin"]})

    def _normalize_theme(self, theme):
        if not theme:
            return ""
        theme = theme.strip()
        known = {
            "fear of authority": "Fear of Authority",
            "fear of negative evaluation & embarrassment": "Fear of Negative Evaluation & Embarrassment",
            "fear of negative evaluation & embarassment": "Fear of Negative Evaluation & Embarrassment",
            "physiological symptoms": "Physiological Symptoms",
            "fear of social gatherings": "Fear of Social Gatherings",
            "fear of strangers & new people": "Fear of Strangers & New People",
            "fear of being observed & performing": "Fear of Being Observed & Performing",
        }
        lowered = theme.lower()
        return known.get(lowered, theme)

    def _resolve_scenario_key(self, theme, raw_key):
        canonical = THEME_SCENARIO_KEYS.get(theme, "general_default")
        allowed = ALLOWED_SCENARIO_KEYS.get(theme, {canonical})
        rk = str(raw_key or "").strip()
        if rk in allowed:
            return rk
        return canonical

    def _scene1_intro(self, theme="", scenario_key=""):
        canonical = THEME_SCENARIO_KEYS.get(theme, "general_default")
        allowed = ALLOWED_SCENARIO_KEYS.get(theme, {canonical})
        sk = scenario_key if scenario_key in allowed else canonical
        setup = {
            "foa_classroom": [
                "**Hati:** You've just walked into your new class. It's the first day for your major subject, so the room is filled with mostly unfamiliar faces.",
                "**Hati:** The room is about half full. Some people are talking quietly, others are on their phones.",
                "**Hati:** There's an empty seat next to a student near the middle of the room. They haven't noticed you yet.",
                "**Hati:** This is your opportunity to practice approaching someone new.",
            ],
            "fsn_classroom": [
                "**Hati:** You've just walked into your new class. It's the first day for your major subject, so the room is filled with students from different sections—mostly unfamiliar faces.",
                "**Hati:** The room is about half full. Some people are talking quietly, others are on their phones.",
                "**Hati:** There's an empty seat next to a student near the middle of the room. They haven't noticed you yet. This is your opportunity to practice approaching someone new.",
                "**Hati:** Before we do anything, let's check in with yourself. This is called a Physical, Intellectual, Emotional, and Social check—it just means noticing what's happening in your body and mind right now.",
            ],
            "foa_supervisor": [
                "**Hati:** You've just entered the department office. The professor is at their desk, talking to another student. They look focused, maybe a bit impatient. You need their signature on your data collection request form. You're waiting for your turn.",
                "**Hati:** Before we go further, let's do a quick Physical, Intellectual, Emotional, and Social check—notice what's happening in your body and mind.",
            ],
            "fne_stage": [
                "**Hati:** You're in a library study room with your group. You've been working on a presentation. Now it's your turn to share your part. You're not 100% ready – maybe you're missing a slide, or your notes aren't complete.",
                "**Hati:** The group looks at you expectantly. Carlo is sitting across from you, arms crossed. Julia is looking at their laptop. Precious gives you a small, encouraging nod.",
                "**Hati:** Before you begin, let's do a Physical, Intellectual, Emotional, and Social check.",
            ],
            "phys_classroom": [
                "**Hati:** You're waiting for the jeep on your route to arrive. Out of nowhere, your heart starts racing. Your palms are sweating. You feel your face flush. Your hands are trembling slightly. There's no clear reason – no threat, no trigger. Just your body doing its own thing.",
                "**Hati:** You're afraid others will notice. A few strangers are nearby, but they seem absorbed in their phones. Then you hear a familiar voice.",
                "**Old Classmate:** Hey! Long time no see! ... Whoa, are you okay? You look kind of nervous.",
                "**Hati:** They noticed. Now you have to respond without saying 'I have anxiety' – because maybe you're not ready to share that. Let's prepare.",
                "**Hati:** Before we do, let's do a Physical, Intellectual, Emotional, and Social check.",
            ],
            "fsg_party": [
                "**Hati:** You've just walked into the party. The birthday friend waves at you from across the room but is immediately pulled into another conversation. You're on your own for now.",
                "**Hati:** Looking around, you see: Group A (three friendly-looking people) - They're laughing, standing near the snacks. One woman makes brief eye contact with you and smiles slightly. The empty corner - A quiet chair with a lamp, away from everyone. Safe, but isolated. Other small groups - People talking, not obviously inviting.",
                "**Hati:** You have a choice: approach the friendly group and introduce yourself, or go to the empty corner. Both are valid, but one might help you practice.",
                "**Hati:** Before you decide, let's do a Physical, Intellectual, Emotional, and Social check.",
            ],
            "fsn_seat": [
                "**Hati:** You've just walked into your usual mall to go for school supplies when you suddenly felt tired. You went to the food hall to rest and order food. It's busier than expected. All the small tables are taken. The only available seats are at that large shared table, where a stranger is sitting. They're focused on their laptop, wearing headphones.",
                "**Hati:** You need a place to sit. You'll have to approach and ask if the seat is free.",
                "**Hati:** Before we do anything, let's do a Physical, Intellectual, Emotional, and Social check.",
            ],
            "fbop_spotlight": [
                "**Hati:** You're standing at the front of the room. Five professors are watching you.",
                "**Hati:** Let me introduce them: Professor 1 (Dr. Reyes) – Middle, neutral expression, taking notes. Professor 2 (Dr. Cruz) – Stern-looking, arms crossed. Professor 3 (Dr. Santos) – Appears supportive, slight nod. Professor 4 (Dr. Garcia) – Silent, staring at laptop. Professor 5 (Dr. Lopez) – Older, seems tired but attentive.",
                "**Hati:** They're taking notes. Some have neutral expressions, some look stern. The room is quiet. You're about to present the objectives of your research.",
                "**Hati:** Before you begin, let's do a Physical, Intellectual, Emotional, and Social check.",
            ],
            "general_default": [
                "**Hati:** You've entered a practice social space.",
                "**Hati:** Let's do a Physical, Intellectual, Emotional, and Social check.",
            ],
        }
        body = setup.get(sk) or setup["general_default"]
        phys_opts = self._pies_physical_options(theme, sk)
        if sk == "phys_classroom":
            phys_q = "**Hati:** How does your body feel right now (in this simulation)?"
        else:
            phys_q = "**Hati:** How does your body feel?"
        messages = list(body) + [phys_q]
        return self._payload(
            messages,
            {"type": "buttons", "options": phys_opts},
        )

    def _pies_physical_options(self, theme, sk):
        if sk == "fsn_classroom":
            return ["Tense", "Relaxed", "Heart racing", "Shaky", "Normal"]
        if theme == "Fear of Being Observed & Performing":
            return ["Tense", "Heart racing", "Shaky", "Sweaty", "Dry mouth", "Normal"]
        if theme == "Physiological Symptoms":
            return ["Heart racing", "Sweating", "Shaking", "Flushed", "Nauseous", "Tense", "Normal"]
        if theme == "Fear of Social Gatherings":
            return ["Tense", "Heart racing", "Shaky", "Sweaty", "Nauseous", "Normal"]
        if theme == "Fear of Negative Evaluation & Embarrassment":
            return ["Tense", "Heart racing", "Shaky", "Sweaty", "Dry mouth", "Nauseous", "Normal"]
        return ["Tense", "Relaxed", "Heart racing", "Shaky", "Sweaty", "Normal"]

    def _pies_emotional(self, data):
        theme = data.get("theme", "")
        sk = data.get("scenario_key", "")
        if theme == "Fear of Strangers & New People" and sk == "fsn_classroom":
            opts = ["Anxious", "Neutral", "Calm", "Irritable", "Sad"]
            return self._payload(
                ["**Hati:** What are you feeling?"],
                {"type": "buttons", "options": opts},
            )
        opts = {
            "Fear of Being Observed & Performing": ["Anxious", "Scared", "Neutral", "Calm", "Irritable", "Overwhelmed"],
            "Physiological Symptoms": ["Embarrassed", "Scared", "Frustrated", "Ashamed", "Neutral", "Overwhelmed"],
            "Fear of Social Gatherings": ["Anxious", "Scared", "Neutral", "Calm", "Irritable", "Overwhelmed", "Lonely"],
            "Fear of Negative Evaluation & Embarrassment": ["Anxious", "Scared", "Neutral", "Calm", "Irritable", "Overwhelmed", "Defensive"],
        }.get(theme, ["Anxious", "Neutral", "Calm", "Irritable", "Scared"])
        return self._payload(
            ["**Hati:** What are you feeling?"],
            {"type": "buttons", "options": opts},
        )

    def _pies_environmental(self, data):
        theme = data.get("theme", "")
        sk = data.get("scenario_key", "")
        opts = []
        if sk == "foa_supervisor":
            opts = ["Professor's stern face", "The other student", "The cluttered desk", "The exit door"]
        elif sk == "fsn_classroom":
            opts = ["People staring", "The empty seat", "The exit door", "The stranger next to the seat"]
        elif sk == "fsn_seat":
            opts = ["The stranger's expression", "The empty seats", "Other people watching", "The exit"]
        elif sk == "fbop_spotlight":
            opts = ["Professors staring", "The clock", "Your notes", "The exit door", "The podium"]
        elif sk == "fsg_party":
            opts = ["The laughing group", "The empty corner", "The exit door", "The birthday friend", "The music"]
        elif sk == "fne_stage":
            opts = ["Carlo's crossed arms", "Julia's laptop", "Precious' nod", "The exit door", "Your notes shaking"]
        elif sk == "phys_classroom":
            opts = ["The classmate's eyes on you", "Other people watching", "Your shaking hands", "The bus arriving", "The exit"]
        else:
            opts = ["People staring", "The empty seat", "The exit door", "The stranger next to the seat"]
        return self._payload(
            ["**Hati:** What do you notice first?"],
            {"type": "buttons", "options": opts},
        )

    def _pies_hati_feedback(self, data):
        phys = (data.get("pies_physical") or "").lower()
        emo = (data.get("pies_emotional") or "").lower()
        theme = data.get("theme", "")
        sk = data.get("scenario_key", "")
        if theme == "Fear of Authority" and sk == "foa_supervisor":
            if phys in ("normal", "relaxed"):
                return "**Hati:** That's great. Calm is a resource. Let's see if we can keep it as we go."
            sym = data.get("pies_physical") or "that"
            return (
                f"**Hati:** I notice you selected {sym}. That's completely normal when facing an authority figure. "
                "Your body is preparing for something important. Let's work with that energy."
            )
        if theme == "Fear of Strangers & New People" and sk == "fsn_classroom":
            if phys in ("normal", "relaxed"):
                return (
                    "**Hati:** That's great that you're feeling calm right now. That's a resource we can use. "
                    "Let's see if we can keep that feeling as we go through this."
                )
            sym = data.get("pies_physical") or "that"
            return (
                f"**Hati:** I notice you selected {sym}. That's completely normal when we're about to do something that feels unfamiliar. "
                "Your body is just getting ready. Let's work with that, not against it."
            )
        if theme == "Fear of Strangers & New People":
            if phys in ("normal", "relaxed"):
                return "**Hati:** Nice. Calm is a superpower here. Let's see if we can keep it."
            sym = data.get("pies_physical") or "that"
            return (
                f"**Hati:** I see you're feeling {sym}. That's your body's way of saying 'this is important.' "
                "It's not dangerous, it's just preparation."
            )
        if theme == "Fear of Being Observed & Performing" and "overwhelmed" in emo:
            return (
                "**Hati:** That's a freeze response. It's okay. We'll take very small steps today. You don't have to be perfect."
            )
        if theme == "Physiological Symptoms" and emo in ("embarrassed", "ashamed"):
            return (
                "**Hati:** Shame is common with visible symptoms. But most people are too busy to notice, and even if they do, they usually forget in minutes."
            )
        if theme == "Fear of Social Gatherings" and "lonely" in emo:
            return "**Hati:** Feeling left out is hard. That's exactly why we're practicing—to build tools for connection."
        if theme == "Fear of Negative Evaluation & Embarrassment" and "defensive" in emo:
            return (
                "**Hati:** You're feeling protective. That's normal when you expect criticism. "
                "Let's channel that energy into clear communication instead of combat."
            )
        tense_like = ["heart racing", "shaky", "tense", "sweaty", "nauseous", "dry mouth", "shaking", "sweating", "flushed"]
        if any(x in phys for x in tense_like):
            sym = data.get("pies_physical") or "that"
            return (
                f"**Hati:** I notice you selected {sym}. That's completely normal when facing something important. "
                "Your body is preparing for something meaningful. Let's work with that energy."
            )
        return "**Hati:** That's great. Calm is a resource. Let's see if we can keep it as we go."

    def _after_pies_environmental(self, data):
        theme = data.get("theme", "")
        sk = data.get("scenario_key", "")
        hati = self._pies_hati_feedback(data)
        if theme == "Fear of Authority" and sk == "foa_supervisor":
            msgs = [
                hati,
                "**Hati:** The other student is almost done. In a moment, you'll need to approach the professor. Let's prepare a simple script so you don't have to think on the spot.",
                "**Hati:** What do you need to say? Here's a basic template. You can use it or make your own:",
            ]
            opts = [
                'A: "Good morning, Professor. I need your signature on this data collection request form for my research."',
                'B: "Excuse me, Professor. Could you please sign this form? It\'s for my thesis data collection."',
                "C: I'll type my own line",
            ]
            return "foa_s2_script", self._payload(msgs, {"type": "buttons", "options": opts})

        if theme == "Fear of Authority" and sk == "foa_classroom":
            msgs = [
                hati,
                "**Hati:** Before you approach that empty seat, think about what you want to happen.",
                "**Hati:** You don't have to become best friends. What would feel like a small win right now?",
            ]
            return "scene2_goal", self._payload(msgs, {"type": "text_input", "placeholder": "Type your goal..."})

        if theme == "Fear of Strangers & New People" and sk == "fsn_classroom":
            msgs = [
                "**Hati:** Before you approach that empty seat, let's think about what you want to happen. You don't have to become best friends with this person. You don't even have to have a conversation. The goal can be much smaller.",
                "**Hati:** What would feel like a small win for you right now? Type what comes to mind.",
            ]
            return "scene2_goal", self._payload(msgs, {"type": "text_input", "placeholder": "e.g. Just sit down without panicking / Maybe say hi / Just exist there"})

        if theme == "Fear of Strangers & New People":
            msgs = [
                hati,
                "**Hati:** Let's keep this simple. You don't need a conversation. You just need to claim a seat. Here's a basic script:",
                "**Hati:** Point to the empty seat and ask: 'Is this seat taken?' That's all. The stranger will probably say 'No' or just shake their head.",
                "**Hati:** Sometimes they might make a small comment, like 'Go ahead' or 'It's free.' If they do, you can just say 'Thanks' and sit down. That's it.",
                "**Hati:** Let's practice your line silently in your head. Ready?",
            ]
            return "fsn_s2_practice", self._payload(msgs, {"type": "buttons", "options": ["Continue"]})

        if theme == "Fear of Being Observed & Performing":
            msgs = [
                hati,
                "**Hati:** First, what is the title of your research? Type it below (even a short version is fine).",
            ]
            return "fbop_s2_title", self._payload(msgs, {"type": "text_input", "placeholder": "Research title..."})

        if theme == "Fear of Social Gatherings":
            msgs = [
                "**Hati:** Let's prepare for either choice. First, I need you to decide your intention for tonight.",
                "**Hati:** Path A: Approach the group. Walk over, say a simple line, maybe exchange a few words. Higher risk, higher reward.",
                "**Hati:** Path B: Take the corner. Sit alone, observe, maybe talk to the birthday friend later. Low risk, safe but isolating.",
                "**Hati:** Which path feels possible for you right now? There's no wrong answer.",
            ]
            opts = ["Path A: Approach the group", "Path B: Take the corner"]
            return "fsg_s2_path", self._payload(msgs, {"type": "buttons", "options": opts})

        if theme == "Fear of Negative Evaluation & Embarrassment":
            msgs = [
                hati,
                "**Hati:** Pick how you'll open when you're not 100% ready—honest, confident, or your own line.",
            ]
            opts = [
                "Honest & Brief: I've got the main points, still working on details—here's what I have.",
                "Confident & Selective: Here's my section—I'll focus on the key findings.",
                "Defensive (not recommended): I'm not done yet, but you wanted me to share, so...",
                "Custom response",
            ]
            return "fne_s2_style", self._payload(msgs, {"type": "buttons", "options": opts})

        if theme == "Physiological Symptoms":
            msgs = [
                hati,
                "**Hati:** Prepare a short answer without revealing anxiety if you're not ready. Pick a line or write your own.",
            ]
            opts = [
                "Oh, I think I had too much coffee. I'm fine.",
                "It's really hot out here, isn't it?",
                "Just one of those days. Thanks for asking.",
                "Yeah, I'm practicing for a marathon—the bus marathon.",
                "Custom response",
            ]
            return "phys_s2_excuse", self._payload(msgs, {"type": "buttons", "options": opts})

        msgs = [hati, "**Hati:** What would feel like a small win right now?"]
        return "scene2_goal", self._payload(msgs, {"type": "text_input", "placeholder": "Type your goal..."})

    def _foa_s2_q_prep_view(self):
        return self._payload(
            [
                "**Hati:** Good. You have your line. Now let's also prepare for possible questions. The professor might ask:",
                "\"What's your research about?\"",
                "\"Have you gotten ethics approval?\"",
                "\"Why do you need this?\"",
                "**Hati:** Think quickly: How would you answer one of those? Just a short sentence.",
            ],
            {"type": "text_input", "placeholder": "Your brief answer..."},
        )

    def _foa_s2_ready_view(self):
        return self._payload(
            [
                "**Hati:** Perfect. You don't need a perfect answer, just something honest. Remember: you're not asking for a favor; this is a standard process. The professor signs forms like this all the time.",
                "**Hati:** Take a breath. Ready?",
            ],
            {"type": "buttons", "options": ["Approach"]},
        )

    def _detected_emotion_lower(self, emotion):
        """Normalize classifier labels from `/scenario/step` or `/scenario/step_audio`."""
        e = (emotion or "").strip().lower()
        if not e:
            return ""
        aliases = {
            "angry": "anger",
            "fearful": "fear",
            "sadness": "sad",
            "joy": "happy",
            "surprised": "surprise",
        }
        return aliases.get(e, e)

    def _foa_s3_npc_view(self):
        return self._payload(
            [
                "**Narrator:** The other student leaves. The professor looks up at you.",
                "**Professor:** Yes? What is it?",
                "**Hati:** Say or type what you say next—the line you prepared, or your own words. For example you might ask for a signature on your data collection form, or say you're nervous but need a signature for your thesis.",
            ],
            {"type": "text_input", "placeholder": "What you say to the professor..."},
        )

    def _foa_story_branch(self, text, emotion):
        t = (text or "").lower()
        emo = self._detected_emotion_lower(emotion)
        if any(x in t for x in ["never mind", "nevermind", "walk away", "i'll go", "i should go", "forget it"]):
            return "freeze"
        if any(x in t for x in ["just sign", "hurry up", "already", "not how we speak"]):
            return "anger"
        clear = any(
            k in t
            for k in [
                "signature",
                "sign this",
                "sign my",
                "data collection",
                "thesis",
                "research",
                "form",
                "professor",
                "good morning",
                "excuse me",
                "please",
            ]
        )
        disfluent = (
            sum(1 for x in ["uh", "um", "...", "i…", "i..."] if x in t) >= 2
            or (("uh" in t or "um" in t) and len(t) < 50 and not clear)
            or "stumbling" in t
        )
        if disfluent:
            return "anxious"
        if clear:
            return "confident"
        if emo in ("anger", "disgust"):
            return "anger"
        if emo in ("sad",):
            return "freeze"
        if emo in ("fear", "anxious"):
            return "anxious"
        if emo in ("happy", "joy", "surprise", "surprised", "neutral", "calm") and len(t) < 200:
            return "confident"
        return "confident"

    def _foa_s3_reaction_enter(self, data):
        data["foa_r_phase"] = 0
        return self._advance_foa_s3_reaction(None, data, None)

    def _advance_foa_s3_reaction(self, state, data, user_text):
        br = data.get("story_branch", "confident")
        phase = int(data.get("foa_r_phase", 0))
        entering = user_text is None

        def _to_scene4():
            if state is not None:
                state["step"] = "scene4_debrief_intro"
            return self._scene4_debrief_intro(data)

        if br == "confident":
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Narrator:** The professor takes your form and glances at it.",
                            "**Professor:** Hmm. What's your research about?",
                            "**Hati:** Answer briefly—you prepared for this.",
                        ],
                        {"type": "text_input", "placeholder": "Your answer in a sentence or two..."},
                    )
                data["foa_research_answer"] = user_text
                data["foa_r_phase"] = 1
                return self._payload(
                    [
                        "**Professor:** Fine. Make sure you have ethics approval. Next.",
                        "**Hati:** See? That was straightforward. You did exactly what you needed to. Notice how your body feels now compared to before.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "anxious":
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Professor:** Speak clearly. What do you need?",
                            "**Hati:** Take a breath. You have your line. Just say it slowly. Say or type what you say next.",
                        ],
                        {"type": "text_input", "placeholder": "What you say next..."},
                    )
                data["foa_retry_pick"] = user_text
                data["foa_r_phase"] = 1
                retry_br = self._foa_story_branch(user_text, data.get("emotion", ""))
                if retry_br == "confident":
                    return self._payload(
                        [
                            "**Professor:** Fine. Make sure you have ethics approval. Next.",
                            "**Hati:** You recovered and spoke clearly. Notice how your body feels now compared to before.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                return self._payload(
                    [
                        "**Professor:** Next time, prepare what you're going to say. Here's your form.",
                        "**Hati:** That felt rough, I know. But look, you still got the signature. The professor wasn't mean, just impatient. You survived. Let's reflect.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "anger":
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Professor:** Excuse me? That's not how we speak to faculty. What is this for?",
                            "**Hati:** Whoa. That tone triggered a defensive response. Let's de-escalate. Apologize briefly and explain—type what you say.",
                        ],
                        {"type": "text_input", "placeholder": "Brief apology and explanation..."},
                    )
                data["foa_apology"] = user_text
                data["foa_r_phase"] = 1
                return self._payload(
                    [
                        "**Professor:** Alright. But remember professionalism matters.",
                        "**Hati:** That was hard, but you recovered. Let's learn from it.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if phase == 0:
            if entering:
                return self._payload(
                    [
                        "**Professor:** Well? Do you need something or not? I don't have all day.",
                        "**Hati:** Hey. You're allowed to pause. Try a short line—for example that you have a form for a signature, or that you need a moment. Or type that you're leaving if you need to step away.",
                    ],
                    {"type": "text_input", "placeholder": "What you say or do next..."},
                )
            data["foa_freeze_pick"] = user_text
            data["foa_r_phase"] = 1
            ut = (user_text or "").lower()
            if any(p in ut for p in ["leave", "avoid", "walk away", "never mind", "nevermind", "i'm leaving", "i need to leave"]):
                return self._payload(
                    [
                        "**Hati:** You left. That's okay—sometimes avoidance feels safer. But let's think: what was the worst that could have happened? Rejection? Annoyance? You can handle those. Next time, let's try a smaller step.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return self._payload(
                [
                    "**Professor:** Fine. Make sure you have ethics approval. Next.",
                    "**Hati:** You did it—with support, but you still spoke. Let's reflect.",
                ],
                {"type": "buttons", "options": ["Continue"]},
            )
        return _to_scene4()

    def _fsn_s2_goal_prompt_view(self):
        return self._payload(
            [
                "**Hati:** Now, what's your small win goal for today? Just sitting down? Asking without stammering? Making eye contact for one second?",
            ],
            {"type": "text_input", "placeholder": "Type your goal, e.g. Just ask and sit"},
        )

    def _fsn_s2_ready_view(self):
        return self._payload(
            [
                "**Hati:** Good. Keep that goal. Take a breath. When you're ready, approach the table.",
            ],
            {"type": "buttons", "options": ["Approach"]},
        )

    def _fsn_s3_npc_view(self):
        return self._payload(
            [
                "**Narrator:** You walk toward the shared table. The stranger looks up briefly, removes one earbud.",
                "**Stranger:** Oh, hey. Need a seat?",
                "**Hati:** Say or type how you respond—for example asking if the seat is taken, if it's okay to sit, or a quick friendly line.",
            ],
            {"type": "text_input", "placeholder": "What you say to the stranger..."},
        )

    def _fsn_story_branch(self, text, emotion):
        t = (text or "").lower()
        emo = self._detected_emotion_lower(emotion)
        if "busy day" in t or "small talk" in t or "how's your" in t:
            return "curious"
        if "whatever" in t or emo in ("anger", "disgust"):
            return "anger"
        if "mumbles" in t or "uh..." in t or "uh " in t or "seat... taken" in t:
            return "anxious"
        if len(t) < 30 and emo in ("fear", "anxious") and not any(
            k in t for k in ["taken", "sit", "free", "okay", "yeah", "yes", "hi", "hey"]
        ):
            return "anxious"
        if "freeze" in t or "nothing" in t or "walk away" in t or "say nothing" in t:
            return "freeze"
        if emo in ("happy", "joy", "surprise", "surprised") and len(t) < 160:
            if any(k in t for k in ("thanks", "great", "nice", "good", "hey", "hi", "hello", "seat", "sit")):
                return "curious"
        if emo in ("fear", "anxious", "sad") and len(t) < 50 and not any(
            k in t for k in ("free", "taken", "sit", "okay", "yeah", "yes")
        ):
            return "anxious"
        return "confident"

    def _fsn_s3_reaction_enter(self, data):
        data["fsn_r_phase"] = 0
        return self._advance_fsn_s3_reaction(None, data, None)

    def _advance_fsn_s3_reaction(self, state, data, user_text):
        br = data.get("story_branch", "confident")
        phase = int(data.get("fsn_r_phase", 0))
        entering = user_text is None

        def _to_scene4():
            if state is not None:
                state["step"] = "scene4_debrief_intro"
            return self._scene4_debrief_intro(data)

        if br == "confident":
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Stranger:** (shakes head, returns to laptop) No, it's free.",
                            "**Hati:** See? That was simple. You asked, they answered. Now just say 'Thanks' and sit down.",
                        ],
                        {"type": "buttons", "options": ["Thanks", "Sit down without speaking"]},
                    )
                data["fsn_seat_ack"] = user_text
                data["fsn_r_phase"] = 1
                return self._payload(
                    [
                        "**Stranger:** (no further interaction, continues working)",
                        "**Hati:** Perfect. You did it. You're sitting. Notice how your body feels now compared to before.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "curious":
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Stranger:** For sure! But anyway, go ahead.",
                            "**Hati:** Nice! You made a tiny connection. That's a bonus. Notice how they responded warmly? That's the norm, not the exception.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
            return _to_scene4()

        if br == "anxious":
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Stranger:** Yeah, go ahead. No problem.",
                            "**Hati:** They didn't judge you. They just said yes. Take a breath and sit. You're okay.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
            return _to_scene4()

        if br == "anger":
            if phase == 1:
                return _to_scene4()
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Stranger:** (looks up, slightly confused) Uh... okay?",
                            "**Hati:** That came off a bit harsh. The stranger didn't do anything wrong. Let's soften it. You can just sit quietly, no need to explain. But notice how anger can push people away.",
                        ],
                        {"type": "buttons", "options": ["Sit down silently", "Continue"]},
                    )
                data["fsn_r_phase"] = 1
                return self._payload(
                    [
                        "**Hati:** Anger often covers fear. If you're feeling scared underneath, that's okay. Let's reflect on that.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "freeze":
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Stranger:** (watches you hesitate, then looks back at laptop)",
                            "**Hati:** Hey, wait. You don't have to leave. You can still ask. Just one word: 'Seat?' Point at the chair. That's enough. Want to try?",
                        ],
                        {
                            "type": "buttons",
                            "options": [
                                "Try: Seat? (point at chair)",
                                "Leave the food hall",
                            ],
                        },
                    )
                ut = (user_text or "").lower()
                data["fsn_freeze_pick"] = user_text
                data["fsn_r_phase"] = 1
                if "leave" in ut:
                    return self._payload(
                        [
                            "**Hati:** You left. That's a valid choice. Sometimes self-protection wins. But let's think: what was the worst that could have happened? A stranger says no? Then you leave anyway. Nothing lost. Next time, let's try just standing near the table for 5 seconds.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                return self._payload(
                    [
                        '**Stranger:** Yeah, sure.',
                        "**Hati:** You did it. You're sitting. Let's reflect.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()
        return _to_scene4()

    def _fbop_s3_outcome_enter(self, data):
        data["fbop_o_phase"] = 0
        return self._advance_fbop_s3_outcome(None, data, None)

    def _advance_fbop_s3_outcome(self, state, data, user_text):
        br = data.get("story_branch", "confident")
        phase = int(data.get("fbop_o_phase", 0))
        entering = user_text is None

        def _to_scene4():
            if state is not None:
                state["step"] = "scene4_debrief_intro"
            return self._scene4_debrief_intro(data)

        def _santos_question_block():
            return self._payload(
                [
                    "**Professor 3 (Dr. Santos, supportive):** Thank you. Can you tell us why these objectives are significant for your field?",
                    "**Hati:** Deliver your prepared answer from Scene 2 — one sentence is enough.",
                ],
                {"type": "text_input", "placeholder": "Type your answer..."},
            )

        def _garcia_followup_block():
            return self._payload(
                [
                    "**Professor 3 (Dr. Santos):** Clear. Thank you.",
                    "**Hati:** Perfect. You handled that smoothly. You're doing well.",
                    "**Professor 4 (Dr. Garcia, silent until now):** Have you considered any limitations to your objectives?",
                    "**Hati:** This is a standard academic question. They're not attacking—they're testing your depth. Answer honestly. It's okay to say 'That's a good point, I will consider that.'",
                ],
                {"type": "text_input", "placeholder": "Type your response..."},
            )

        if br == "confident":
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Narrator:** Dr. Reyes nods. Dr. Santos smiles slightly. Dr. Cruz remains stern but takes notes.",
                            "**Narrator:** Dr. Garcia looks up briefly. Dr. Lopez listens.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                data["fbop_o_phase"] = 1
                return _santos_question_block()
            if phase == 1:
                data["fbop_significance_answer"] = user_text
                data["fbop_o_phase"] = 2
                return _garcia_followup_block()
            if phase == 2:
                data["fbop_limitation_answer"] = user_text
                data["fbop_o_phase"] = 3
                return self._payload(
                    [
                        "**Professor 4 (Dr. Garcia):** Alright. Thank you.",
                        "**Hati:** Good. You handled an unexpected question. That's real progress.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "anxious":
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Professor 2 (Dr. Cruz, stern):** Speak up, please. We can't hear you.",
                            "**Hati:** Take a breath. They're not angry, they just need to hear you. Say or type what you do next—for example Of course and then speaking louder, apologizing for nerves, or if you freeze, type that you froze and Hati will help you read from your notes.",
                        ],
                        {"type": "text_input", "placeholder": "What you say or do next..."},
                    )
                ut = (user_text or "").lower()
                data["fbop_anxious_pick"] = user_text
                if (
                    not ut.strip()
                    or "freeze" in ut
                    or "say nothing" in ut
                    or "can't speak" in ut
                    or "froze" in ut
                    or "a3" in ut
                ):
                    data["fbop_o_phase"] = 13
                    return self._payload(
                        [
                            "**Hati:** It's okay to pause. Just read the next sentence from your notes. Word for word.",
                        ],
                        {"type": "text_input", "placeholder": "Type the next sentence from your notes..."},
                    )
                data["fbop_o_phase"] = 1
                return self._payload(
                    [
                        "**Narrator:** Dr. Cruz nods once. You continue.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            if phase == 13:
                data["fbop_notes_line"] = user_text
                data["fbop_o_phase"] = 14
                return self._payload(
                    [
                        "**Hati:** You found a way through. Let's debrief.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            if phase == 14:
                return _to_scene4()
            if phase == 1:
                data["fbop_o_phase"] = 2
                return _santos_question_block()
            if phase == 2:
                data["fbop_significance_answer"] = user_text
                data["fbop_o_phase"] = 3
                return _garcia_followup_block()
            if phase == 3:
                data["fbop_limitation_answer"] = user_text
                data["fbop_o_phase"] = 4
                return self._payload(
                    [
                        "**Professor 4 (Dr. Garcia):** Alright. Thank you.",
                        "**Hati:** Good. You handled an unexpected question. That's real progress.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            if phase == 4:
                return _to_scene4()
            return _to_scene4()

        if br == "anger":
            if phase == 1:
                return _to_scene4()
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Professor 2 (Dr. Cruz, stern):** Excuse me? We're here to help you. There's no need for that attitude.",
                            "**Hati:** Defensiveness will hurt you. You can recover. Say or type what you do next—a brief apology and reset, pushing back, staying silent, or anything honest.",
                        ],
                        {"type": "text_input", "placeholder": "What you say next..."},
                    )
                ut = (user_text or "").lower()
                data["fbop_anger_pick"] = user_text
                if any(
                    k in ut
                    for k in ["sorry", "apolog", "nervous", "didn't mean", "restat", "let me explain", "my mistake"]
                ):
                    data["fbop_o_phase"] = 1
                    return self._payload(
                        [
                            "**Narrator:** Dr. Cruz relaxes slightly. It's fine. Continue.",
                            "**Hati:** You recovered. Let's debrief.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                if any(k in ut for k in ["not defensive", "wasn't rude", "you're being", "c2", "double"]):
                    data["fbop_o_phase"] = 1
                    return self._payload(
                        [
                            "**Dr. Cruz:** Perhaps you should step out and compose yourself.",
                            "**Hati:** Let's pause the simulation here and debrief.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                if not ut.strip() or "say nothing" in ut or ut.strip() in (".", "...", "—") or "c3" in ut:
                    data["fbop_o_phase"] = 1
                    return self._payload(
                        [
                            "**Dr. Reyes:** Would you like a moment?",
                            "**Hati:** Silence is okay—let's reset next time with one breath and one sentence.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                data["fbop_o_phase"] = 1
                return self._payload(
                    [
                        "**Dr. Reyes:** Would you like a moment?",
                        "**Hati:** Thank you for answering honestly. Let's debrief.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "freeze":
            if phase == 18:
                return _to_scene4()
            if phase == 0:
                if entering:
                    return self._payload(
                        [
                            "**Professor 1 (Dr. Reyes, neutral but noticing)**: Take your time. We're not going anywhere.",
                            "**Hati:** You're freezing. That's a normal fear response. You don't have to leave. Say or type what you do next—for example okay, that you need a moment, that you're leaving, or your own words.",
                        ],
                        {"type": "text_input", "placeholder": "What you say or do next..."},
                    )
                ut = (user_text or "").lower()
                data["fbop_freeze_pick"] = user_text
                if any(k in ut for k in ["exit", "leave", "walk out", "out the door", "d3", "can't stay"]):
                    data["fbop_o_phase"] = 30
                    return self._payload(
                        [
                            "**Hati:** You left. That's okay. Let's debrief.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                if any(k in ut for k in ["moment", "minute", "pause", "aside", "d2"]):
                    data["fbop_o_phase"] = 18
                    return self._payload(
                        [
                            "**Narrator:** The professors wait quietly.",
                            "**Hati:** Take 30 seconds. Then come back to the podium.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                data["fbop_o_phase"] = 1
                return self._payload(
                    [
                        "**Dr. Reyes:** Whenever you're ready.",
                        "**Hati:** Now read your first objective. Just the first one.",
                    ],
                    {"type": "text_input", "placeholder": "Type your first objective..."},
                )
            if phase == 15:
                data["fbop_freeze_custom"] = user_text
                data["fbop_o_phase"] = 22
                return self._payload(["**Hati:** Thank you. Let's debrief."], {"type": "buttons", "options": ["Continue"]})
            if phase == 22:
                return _to_scene4()
            if phase == 30:
                return _to_scene4()
            if phase == 1:
                data["fbop_first_objective"] = user_text
                data["fbop_o_phase"] = 17
                return self._payload(
                    [
                        "**Hati:** You stayed in the room and used your notes. That's a win to build on.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            if phase == 17:
                return _to_scene4()
        return _to_scene4()

    def _fbop_s2_opening_view(self, data):
        title = data.get("fbop_title", "[your title]")
        return self._payload(
            [
                "**Hati:** Good. Now choose an opening line (or custom).",
                f'A. "Good morning, esteemed panel. Today I will present the objectives of my research titled \'{title}\'."',
                f'B. "Hello. My research is about this topic—here are my objectives." (title: {title})',
                'C. "Thank you for being here. My objectives are: first..."',
                "Custom opening",
            ],
            {"type": "buttons", "options": ["Use option A (formal)", "Use option B (simple)", "Use option C (minimal)", "Custom opening"]},
        )

    def _fbop_s2_pause_view(self):
        return self._payload(
            [
                "**Hati:** Great. Now practice saying that opening line silently in your head three times. Then take a breath.",
            ],
            {"type": "buttons", "options": ["Continue"]},
        )

    def _fbop_s2_contrib_view(self):
        return self._payload(
            [
                "**Hati:** Now, let's prepare for what comes after. After you state your objectives, one professor will likely ask a question. The most common question is: 'Why are these objectives significant?'",
                "**Hati:** Let's prepare a short answer. What is the main contribution of your research? Type one sentence.",
            ],
            {"type": "text_input", "placeholder": "One sentence on significance..."},
        )

    def _fbop_s2_goal_view(self):
        return self._payload(
            [
                "**Hati:** Pick a small win goal for this presentation:",
            ],
            {
                "type": "buttons",
                "options": [
                    "Speak the first sentence without stopping",
                    "Make eye contact with one professor",
                    "Read objectives from notes without apologizing",
                    "Just get through without leaving the room",
                    "Custom goal",
                ],
            },
        )

    def _fbop_s2_ground_view(self):
        return self._payload(
            [
                "**Hati:** Quick grounding before you start—pick one:",
            ],
            {
                "type": "buttons",
                "options": [
                    "Feet on floor: press flat, feel the ground",
                    "Back against the chair: feel support",
                    "Hand on chest: notice your heartbeat",
                    "One deep breath: in 4, hold 2, out 6",
                ],
            },
        )

    def _fbop_s2_ready_view(self):
        return self._payload(
            [
                "**Hati:** They're evaluators, not enemies. Their job is to ask questions, not to humiliate you.",
                "**Hati:** When you're ready, tap Begin Presentation.",
            ],
            {"type": "buttons", "options": ["Begin Presentation"]},
        )

    def _fbop_s3_delivery_view(self):
        return self._payload(
            [
                "**Hati:** You're at the podium. Five professors watch.",
                "**Hati:** Say or type your opening line—the one you prepared",
            ],
            {"type": "text_input", "placeholder": "Your opening line or how you're delivering it..."},
        )

    def _fbop_story_branch(self, text, emotion):
        t = (text or "").lower()
        emo = self._detected_emotion_lower(emotion)
        if "defensive" in t or "angry" in t or "attitude" in t or "just read" in t:
            return "anger"
        if "barely" in t or "mumbling" in t or "mumble" in t or "too quiet" in t:
            return "freeze"
        if "quiet" in t or "hesitant" in t or "shaky" in t or "nervous" in t or "rushed" in t or "fast" in t:
            return "anxious"
        if "good morning" in t or "thank you" in t or "objectives" in t or "present" in t:
            return "confident"
        if emo in ("anger", "disgust") and len(t) < 400:
            return "anger"
        if emo in ("fear", "anxious") and len(t) < 400:
            return "anxious"
        if emo in ("sad",) and len(t) < 300:
            return "freeze"
        if emo in ("happy", "joy", "surprise", "surprised", "neutral", "calm") and len(t) < 500:
            return "confident"
        return "confident"

    def _fsg_a_opening_view(self, data):
        n = (data.get("user_name") or "there").strip() or "there"
        return self._payload(
            [
                "**Hati:** Great. Let's prepare a simple opening line. You don't need to make it a long conversation, just a few words.",
            ],
            {
                "type": "buttons",
                "options": [
                    f'Hey, is this seat taken? I\'m {n}, by the way.',
                    "Hi, I don't know many people here. Mind if I join you?",
                    f"Hi, I'm {n}. The birthday friend said I should come say hi.",
                    "Custom response",
                ],
            },
        )

    def _fsg_a_practice_pause_view(self):
        return self._payload(
            [
                "**Hati:** Good. Now practice that line silently in your head three times.",
            ],
            {"type": "buttons", "options": ["Continue"]},
        )

    def _fsg_a_relation_view(self):
        return self._payload(
            [
                "**Hati:** Now, let's anticipate their likely response. Most people at parties are friendly. They'll probably say something like: 'Oh, hey! Are you here for the birthday?' or 'Of course, sit down! How do you know the birthday person?'",
                "**Hati:** Let's prepare a follow-up answer. How do you know the birthday friend?",
            ],
            {"type": "text_input", "placeholder": "e.g. We're classmates..."},
        )

    def _fsg_a_goal_view(self):
        return self._payload(
            [
                "**Hati:** Perfect, now set your small win goal for approaching",
            ],
            {
                "type": "buttons",
                "options": [
                    "Say your opening line without apologizing",
                    "Make eye contact with one person",
                    "Stay at least 30 seconds before leaving",
                    "Ask one follow-up question",
                    "Custom goal",
                ],
            },
        )

    def _fsg_b_goal_view(self):
        return self._payload(
            [
                "**Hati:** From the corner you can still try a micro-step. Pick a goal:",
            ],
            {
                "type": "buttons",
                "options": [
                    "Sit in the corner 5 minutes without leaving",
                    "Make eye contact and smile at one person",
                    "When the host comes over, ask to meet one new person",
                    "Custom goal",
                ],
            },
        )

    def _fsg_ground_view(self):
        return self._payload(
            [
                "**Hati:** Now, let's do a quick grounding exercise before you start. I want you to do one of these:",
            ],
            {
                "type": "buttons",
                "options": [
                    "Feet on floor",
                    "One deep breath (in 4, hold 2, out 6)",
                    "Hand on heart—feel your heartbeat",
                    "Name three objects you see",
                ],
            },
        )

    def _fsg_s2_proceed_view(self):
        return self._payload(
            ["**Hati:** Good. You're present. Tap Proceed when ready to move in the room."],
            {"type": "buttons", "options": ["Proceed"]},
        )

    def _fsg_s3_social_view(self, data):
        path = data.get("fsg_path", "")
        if "path b" in path.lower() or "corner" in path.lower():
            return self._payload(
                [
                    "**Hati:** You're seated away from the group.",
                    "**Hati:** Can you make brief eye contact and a small smile toward the group?",
                ],
                {
                    "type": "buttons",
                    "options": [
                        "Yes—eye contact + small smile",
                        "Stay quiet on my phone",
                        "Head over to approach the group now",
                    ],
                },
            )
        prepared = (data.get("fsg_open_custom") or data.get("fsg_open_pick") or "").strip()
        lines = [
            "**Narrator:** You walk toward the group. Julia turns slightly.",
            '**Julia:** Oh, hey! Are you here for the birthday?',
            "**Hati:** Say your opening line now—just as you practiced. You can type it here or use the microphone.",
        ]
        if prepared:
            lines.append(f"Hati: Your prepared opening: {prepared}")
        return self._payload(
            lines,
            {"type": "text_input", "placeholder": "Type or dictate your opening line..."},
        )

    def _fsg_branch_from_opening_delivery(self, text, detected_emotion, pies_emotional=None):
        """Map user's opening to Scene 3 branch. Text cues first; classifier + optional PIES nudge when ambiguous."""
        t = (text or "").strip().lower()
        emo_d = self._detected_emotion_lower(detected_emotion)
        emo_p = self._detected_emotion_lower(pies_emotional)
        ctx = emo_d or emo_p
        if not t or t in (".", "...", "-", "leave", "never mind", "nevermind", "nothing"):
            return "freeze"
        if any(x in t for x in ["whatever", "not my problem", "ugh", "who cares", "annoying", "go away", "shut up"]):
            return "anger"
        if any(
            phrase in t
            for phrase in (
                "busy day",
                "nice party",
                "love the music",
                "great music",
                "how do you know",
                "how's your night",
            )
        ):
            return "curious"
        stammer = t.startswith("uh") or t.startswith("um") or "uh..." in t or "um..." in t
        apologetic = " sorry " in f" {t} " or t.startswith("sorry") or "i don't know" in t or "i'm not sure" in t
        if stammer or apologetic:
            return "anxious"
        if len(t) < 10 and "?" not in t:
            return "anxious"
        if emo_d in ("anger", "disgust") or emo_p in ("irritable", "anger", "disgust"):
            if len(t) < 50:
                return "anger"
        if emo_d in ("fear", "anxious") or emo_p in ("fear", "anxious", "scared", "overwhelmed", "sad"):
            if len(t) < 45:
                return "anxious"
        if emo_d in ("happy", "joy", "surprise", "surprised") and len(t) >= 8:
            if any(k in t for k in ("hey", "hi", "hello", "birthday", "party", "nice", "great", "thanks")):
                return "curious"
        return "confident"

    def _fsg_story_branch(self, text, emotion, path):
        t = (text or "").lower()
        emo = self._detected_emotion_lower(emotion)
        if "phone" in t or "quiet" in t:
            return "avoid"
        if "approach" in t and "corner" in path.lower():
            return "confident"
        if "irritated" in t or emo in ("anger", "disgust"):
            return "anger"
        if "freeze" in t or "nothing" in t:
            return "freeze"
        if "hesitant" in t or emo in ("fear", "anxious"):
            return "anxious"
        if emo in ("sad",) and len(t) < 60:
            return "anxious"
        if "eye contact" in t and "path b" in path.lower():
            return "confident"
        return "confident"

    def _advance_fsg_s3_reaction(self, state, data, user_text):
        """Party Scene 3 — follow script beats: follow-up answers, corner choices, opening flow from corner."""
        path_raw = data.get("fsg_path") or ""
        path_l = path_raw.lower()
        is_corner = "path b" in path_l or "corner" in path_l
        br = data.get("story_branch", "confident")
        ph = int(data.get("fsg_rx_ph", 0))
        social_l = ((data.get("fsg_social_pick") or "") + "").lower()
        opening_flow = bool(data.get("fsg_opening_flow")) and is_corner
        entering = user_text is None

        def _to_scene4():
            if state is not None:
                state["step"] = "scene4_debrief_intro"
            for k in ("fsg_rx_ph", "fsg_opening_flow"):
                data.pop(k, None)
            return self._scene4_debrief_intro(data)

        def _opening_from_corner_payload():
            prepared = (data.get("fsg_open_custom") or data.get("fsg_open_pick") or "").strip()
            lines = [
                "**Narrator:** You walk toward the group. Julia turns slightly.",
                '**Julia:** Oh, hey! Are you here for the birthday?',
                "**Hati:** Say your opening line now—just as you practiced. You can type it here or use the microphone.",
            ]
            if prepared:
                lines.append(f"Hati: Your prepared opening: {prepared}")
            return self._payload(
                lines,
                {"type": "text_input", "placeholder": "Type or dictate your opening line..."},
            )

        if entering:
            data["fsg_rx_ph"] = 0
            if opening_flow:
                return _opening_from_corner_payload()
            if is_corner and br == "confident":
                return self._payload(
                    [
                        "**Hati:** Julia notices and smiles back, then returns to the conversation.",
                        "**Hati:** Nice. They smiled back. That's a tiny connection. Now, you could stay here, or you could use that as an invitation to approach. What do you want to do?",
                    ],
                    {
                        "type": "buttons",
                        "options": [
                            "Stay in corner (satisfied with smile)",
                            "Approach group now",
                            "Wait for birthday friend",
                        ],
                    },
                )
            if is_corner and br == "avoid":
                return self._payload(
                    [
                        "**Hati:** You're avoiding. That's a choice. But let's be honest: will you feel better leaving without trying, or would you feel a little proud if you made one small effort?",
                    ],
                    {
                        "type": "buttons",
                        "options": [
                            "I'll try eye contact and a small smile",
                            "I want to stay on my phone / leave soon",
                        ],
                    },
                )
            if is_corner:
                return self._payload(
                    [
                        "**Hati:** Avoidance is a choice. Try 10 minutes before leaving—or one smile next time.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            if br == "curious":
                return self._payload(
                    [
                        "**Julia:** Yeah totally—pull up a chair! This is Jaspher and Precious.",
                        "**Jaspher:** How do you know the birthday friend?",
                        "**Hati:** Nice opening—they're including you. Answer with how you know the host—keep it short.",
                    ],
                    {"type": "text_input", "placeholder": "How do you know the birthday friend? (one line)"},
                )
            if br == "confident":
                return self._payload(
                    [
                        "**Julia:** Nice to meet you! This is Jaspher and Precious. Pull up a chair!",
                        "**Jaspher:** How do you know the birthday friend?",
                        "**Hati:** They're including you. Answer with the relationship you prepared—just a short line.",
                    ],
                    {"type": "text_input", "placeholder": "How do you know the birthday friend? (one line)"},
                )
            if br == "anxious":
                return self._payload(
                    [
                        "**Julia:** Hey, no worries. Want to sit down? I'm Julia.",
                        "**Precious:** Yeah, grab a seat. We don't bite.",
                        "**Jaspher:** So, how do you know the birthday friend?",
                        "**Hati:** They're being kind. They can tell you're nervous, and they're not judging. Give a small answer—even a few words is enough.",
                    ],
                    {"type": "text_input", "placeholder": "Your answer (even a few words)..."},
                )
            if br == "anger":
                return self._payload(
                    [
                        "**Julia:** Oh... okay. Well, you can sit if you want.",
                        "**Jaspher:** exchanges a glance with Precious.",
                        "**Hati:** That came off as hostile. They didn't do anything wrong. Let's soften. Say: 'Sorry, I'm just nervous. Thanks for being nice.'",
                    ],
                    {
                        "type": "buttons",
                        "options": [
                            "Sorry, I'm just nervous. Thanks for being nice.",
                            "Stay quiet / say nothing more",
                        ],
                    },
                )
            return self._payload(
                [
                    "**Julia:** Hey, you okay? You can sit if you want.",
                    "**Hati:** Don't leave yet. Just pause. Turn around and say one word: 'Okay.' Then take a breath.",
                ],
                {
                    "type": "buttons",
                    "options": [
                        "Say: Okay (then sit down quietly)",
                        "Leave the party",
                    ],
                },
            )

        ut = (user_text or "").lower()

        if is_corner and opening_flow:
            if ph == 0:
                data["fsg_delivered_opening"] = user_text
                data["fsg_rx_ph"] = 1
                return self._payload(
                    [
                        "**Precious:** Cool.",
                        "**Precious:** So, what do you think of the music?",
                        "**Hati:** See? They're including you. You don't need to be clever—just answer naturally.",
                    ],
                    {"type": "text_input", "placeholder": "Your answer about the music (one line)..."},
                )
            if ph == 1:
                data["fsg_party_music"] = user_text
                data["fsg_rx_ph"] = 2
                return self._payload(
                    [
                        "**Julia:** laughs: Yeah, this is great for loosening up.",
                        "**Hati:** You've been in the conversation for over a minute. That's a win. Notice how your body feels now compared to before.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if is_corner and br == "confident" and ph == 0 and not opening_flow:
            if "stay" in ut:
                return _to_scene4()
            if "wait" in ut:
                data["fsg_rx_ph"] = 10
                return self._payload(
                    [
                        "**Hati:** Okay. Let's wait. But set a timer for 10 minutes if you can—that keeps you from disappearing all night.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            if "approach" in ut:
                data["fsg_opening_flow"] = True
                data["fsg_rx_ph"] = 0
                return _opening_from_corner_payload()

        if is_corner and br == "confident" and ph == 10:
            return _to_scene4()

        if is_corner and br == "avoid" and ph == 0:
            if "eye contact" in ut or "smile" in ut:
                data["story_branch"] = "confident"
                data["fsg_rx_ph"] = 0
                return self._advance_fsg_s3_reaction(state, data, None)
            data["fsg_rx_ph"] = 20
            return self._payload(
                [
                    "**Hati:** Okay. Let's at least stay for 10 minutes so you're not running away. Then you can leave.",
                ],
                {"type": "buttons", "options": ["Continue"]},
            )

        if is_corner and br == "avoid" and ph == 20:
            return _to_scene4()

        if is_corner:
            return _to_scene4()

        if br in ("confident", "curious"):
            if ph == 0:
                data["fsg_party_relation"] = user_text
                data["fsg_rx_ph"] = 1
                return self._payload(
                    [
                        "**Precious:** Cool.",
                        "**Precious:** So, what do you think of the music?",
                        "**Hati:** They're including you. You don't need to be clever—just answer naturally.",
                    ],
                    {"type": "text_input", "placeholder": "Your answer about the music (one line)..."},
                )
            if ph == 1:
                data["fsg_party_music"] = user_text
                data["fsg_rx_ph"] = 2
                return self._payload(
                    [
                        "**Julia:** laughs: Yeah, this is great for loosening up.",
                        "**Hati:** You've been in the conversation for over a minute. That's a win. Notice how your body feels now compared to before.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "anxious":
            if ph == 0:
                data["fsg_party_relation"] = user_text
                data["fsg_rx_ph"] = 1
                return self._payload(
                    [
                        "**Julia:** Cool. Well, you're welcome here. Take your time.",
                        "**Hati:** They accepted you even when you were quiet. That's important evidence: people don't require you to be perfect.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "anger":
            if ph == 0:
                if "sorry" in ut or "nervous" in ut:
                    data["fsg_rx_ph"] = 1
                    return self._payload(
                        [
                            "**Julia:** No worries! We've all been there. Sit down.",
                            "**Hati:** Repair lands. You're still learning the tone that matches your intention.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                data["fsg_rx_ph"] = 1
                return self._payload(
                    [
                        "**Precious:** Well, we're gonna grab drinks. See you around.",
                        "**Hati:** That felt like rejection—and it stings. Notice they moved on, not because you're worthless, but because the vibe got prickly. Next time, try a softer start.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "freeze":
            if ph == 0:
                if "leave" in ut:
                    return self._payload(
                        [
                            "**Hati:** You left. That's okay. Let's debrief why.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                return self._payload(
                    [
                        "**Hati:** You sit down quietly.",
                        "**Hati:** You did it. You're sitting. Just stay for 30 seconds if you can—that counts.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        return _to_scene4()

    def _fne_s2_pause_view(self):
        return self._payload(
            [
                "**Hati:** Good. Now practice that opening line silently in your head.",
                "**Hati:** Take a breath when you're ready for the next step.",
            ],
            {"type": "buttons", "options": ["Continue"]},
        )

    def _fne_s2_points_view(self):
        return self._payload(
            [
                "**Hati:** Now, let's anticipate what you'll actually share. What are 1–2 key points you can say even if you're not fully ready?",
            ],
            {"type": "text_input", "placeholder": "Key points..."},
        )

    def _fne_s2_goal_view(self):
        return self._payload(
            [
                "**Hati:** Small win goal for this meeting:",
            ],
            {
                "type": "buttons",
                "options": [
                    "Say my part without apologizing for being unprepared",
                    "Make eye contact with Precious while speaking",
                    "After Carlo's criticism, say one calm response",
                    "Don't leave the room",
                    "Custom goal",
                ],
            },
        )

    def _fne_s2_ground_view(self):
        return self._payload(
            [
                "**Hati:** Grounding—pick one:",
            ],
            {
                "type": "buttons",
                "options": [
                    "Feet on floor",
                    "One deep breath (in 4, hold 2, out 6)",
                    "Hand on chest",
                    "Name three things in the room",
                ],
            },
        )

    def _fne_s2_ready_view(self):
        return self._payload(
            [
                "**Hati:** Your worth isn't determined by one person's frown. Tap when ready to share your part.",
            ],
            {"type": "buttons", "options": ["Share Your Part"]},
        )

    def _fne_s3_carlo_view(self):
        return self._payload(
            [
                "**Hati:** You share your opening and key points. Precious nods. Julia glances up.",
                "**Carlo:** That's it?",
                "**Hati:** Ouch—but notice: dismissive, not 'your work is trash.' How do you respond? Say or type what you say next.",
            ],
            {"type": "text_input", "placeholder": "Your response to Carlo..."},
        )

    def _fne_story_branch(self, text, emotion):
        """
        Branch Carlo follow-up: typed cues first, then system-detected emotion when text is ambiguous
        (e.g. same line read as calm vs anxious vs upbeat).
        """
        t = (text or "").lower()
        emo = self._detected_emotion_lower(emotion)

        if not t.strip() or "freeze" in t or "say nothing" in t or t.strip() in (".", "...", "—"):
            return "freeze"

        if "back off" in t or "shut up" in t or "leave me alone" in t:
            return "anger"
        if emo in ("anger", "disgust") and "sorry" not in t:
            return "anger"

        if "sorry" in t or "not enough" in t or "i'll do more" in t or "i know it's" in t:
            return "apologetic"

        if "open to" in t or "what else" in t or "suggestions" in t:
            return "curious"

        if any(
            k in t
            for k in (
                "ready",
                "draft",
                "detail",
                "friday",
                "move on",
                "boundary",
                "what i have",
                "have more",
                "full version",
            )
        ):
            return "calm"

        if emo in ("fear", "anxious") and len(t) < 220:
            return "apologetic"
        if emo in ("happy", "joy", "surprise", "surprised") and len(t) < 220:
            return "curious"
        if emo == "sad" and len(t) < 120:
            return "freeze"
        if emo in ("neutral", "calm") and len(t) < 220:
            return "calm"

        return "calm"

    def _fne_s3_outcome_view(self, data):
        br = data.get("story_branch", "calm")
        if br == "calm":
            msgs = [
                "**Carlo:** Alright. Just checking. Precious: Good start. Julia: We can fill gaps later.",
                "**Hati:** You didn't collapse or attack—you stated your boundary.",
            ]
        elif br == "curious":
            msgs = [
                "**Carlo:** I thought you'd have more data—okay, maybe we brainstorm. Precious: Let's help each other.",
                "**Hati:** You turned criticism into collaboration.",
            ]
        elif br == "apologetic":
            return self._payload(
                [
                    "**Carlo:** Yeah, you should. We can't present that.",
                    "**Precious:** Hey, it's fine. We have time. Let's not pressure them.",
                    "**Hati:** Apologizing for being unprepared made you look weaker, and Carlo pounced. But you can recover. Say or type what you say next—for example that you'll have the full version by Friday and want to move on, or another honest line.",
                ],
                {"type": "text_input", "placeholder": "Your recovery line..."},
            )
        elif br == "anger":
            msgs = [
                "**Carlo:** Excuse me? I'm just asking. Precious: Let's calm down. Julia: Maybe a break.",
                "**Hati:** Anger escalated things—brief repair helps: 'Sorry, I'm stressed—here's what I have.'",
            ]
        else:
            msgs = [
                "**Precious:** It's okay—we're all stressed. Want to share more later?",
                "**Hati:** Freezing happens. A simple 'I'll email the rest' buys time.",
            ]
        return self._payload(msgs, {"type": "buttons", "options": ["Continue"]})

    def _phys_s2_ack_view(self):
        return self._payload(
            [
                "**Hati:** Now, your classmate might follow up with:",
                "\"You sure? You look really pale.\"",
                "\"Want to sit down?\"",
                "\"Is something going on?\"",
                "**Hati:** Here's a safe follow-up script: 'I appreciate you asking, but I'm okay. Just need to catch my breath.'",
            ],
            {"type": "buttons", "options": ["I've got it—thanks, Hati"]},
        )

    def _phys_s2_goal_view(self):
        return self._payload(
            [
                "**Hati:** Now set your small win goal for this interaction:",
            ],
            {
                "type": "buttons",
                "options": [
                    "Say my excuse without apologizing for my symptoms",
                    "Make eye contact for just one second",
                    "Change the subject (e.g., 'So, what have you been up to?')",
                    "Stay at the bus stop (don't run away)",
                    "Custom goal",
                ],
            },
        )

    def _phys_s2_ground_view(self):
        return self._payload(
            [
                "**Hati:** Let's do a quick grounding exercise to lower your physiological arousal right now:",
            ],
            {
                "type": "buttons",
                "options": [
                    "Slow breathing: in 4, hold 2, out 6 (x3)",
                    "Cold water on wrists / face (simulate)",
                    "Tense fists 5s, then release",
                    "5-4-3-2-1 senses",
                ],
            },
        )

    def _phys_s2_ready_view(self):
        return self._payload(
            [
                "**Hati:** Symptoms feel huge to you; most people barely clock them. Tap when ready to respond.",
            ],
            {"type": "buttons", "options": ["Respond to classmate"]},
        )

    def _phys_s3_classmate_view(self):
        return self._payload(
            [
                "**Old classmate:** Hey! Long time no see! ... Whoa, are you okay? You look kind of nervous.",
                "**Hati:** Use your prepared line—or choose a style:",
            ],
            {
                "type": "buttons",
                "options": [
                    "Calm, casual, believable line",
                    "Anxious, unconvincing line",
                    "Defensive / irritated reply",
                    "Freeze—can't speak",
                    "Custom response",
                ],
            },
        )

    def _phys_story_branch(self, text, emotion):
        t = (text or "").lower()
        emo = self._detected_emotion_lower(emotion)
        if "freeze" in t:
            return "freeze"
        if "defensive" in t or "irritated" in t or emo in ("anger", "disgust"):
            return "anger"
        if "anxious" in t or "unconvincing" in t or emo in ("fear", "anxious"):
            return "anxious"
        if emo == "sad" and len(t) < 120:
            return "anxious"
        if emo in ("happy", "joy", "surprise", "surprised", "neutral", "calm") and len(t) < 200:
            return "calm"
        return "calm"

    def _advance_phys_s3_reaction(self, state, data, user_text):
        br = data.get("story_branch", "calm")
        ph = int(data.get("phys_r_ph", 0))

        def _to_scene4():
            if state is not None:
                state["step"] = "scene4_debrief_intro"
            data.pop("phys_r_ph", None)
            return self._scene4_debrief_intro(data)

        if user_text is None:
            data["phys_r_ph"] = 0
            if br == "calm":
                return self._payload(
                    [
                        "**Classmate:** Oh okay, cool. Yeah, coffee does that to me too. Anyway, good to see you! How's school?",
                        "**Hati:** See? They bought it. Or they're being polite. Either way, the moment passed. Now you can change the subject.",
                    ],
                    {"type": "text_input", "placeholder": "Answer briefly or change the subject (one line)..."},
                )
            if br == "anxious":
                return self._payload(
                    [
                        "**Classmate:** (more concerned) You sure? You don't look fine. Want to sit down for a minute? The bus isn't here yet.",
                        "**Hati:** They're not judging—they're worried. That's kindness. You can accept the offer without revealing anxiety.",
                    ],
                    {
                        "type": "buttons",
                        "options": [
                            "Thanks—I'll sit for a minute.",
                            "I'm okay, really. I'll just stand for a sec—thanks though.",
                        ],
                    },
                )
            if br == "anger":
                return self._payload(
                    [
                        "**Classmate:** (taken aback) Whoa, sorry I asked. Just trying to help.",
                        "**Hati:** Defensiveness pushes people away. They were being kind. You can recover: 'Sorry, I'm just stressed. Thanks for asking.'",
                    ],
                    {
                        "type": "buttons",
                        "options": [
                            "Sorry, I'm just stressed. Thanks for asking.",
                            "Stay defensive / say nothing",
                        ],
                    },
                )
            return self._payload(
                [
                    "**Classmate:** ... Hello? You okay? Should I call someone?",
                    "**Hati:** Say anything. Just one word: 'Yeah.' Then take a breath.",
                ],
                {
                    "type": "buttons",
                    "options": [
                        "Yeah",
                        "Stay frozen—can't speak",
                    ],
                },
            )

        ut = (user_text or "").lower()

        if br == "calm":
            if ph == 0:
                data["phys_calm_chitchat"] = user_text
                data["phys_r_ph"] = 1
                return self._payload(
                    [
                        "**Classmate:** Cool. Well, I\'ll take my leave now. Take care, okay?",
                        "**Hati:** You handled that smoothly. You protected your privacy and kept the interaction positive.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "anxious":
            if ph == 0:
                data["phys_anx_sit_pick"] = user_text
                data["phys_r_ph"] = 1
                if "sit" in ut:
                    return self._payload(
                        [
                            "**Narrator:** You sit for a moment. Your classmate stays nearby.",
                            "**Classmate:** Take your time.",
                            "**Hati:** You're allowing help. That's strength.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                return self._payload(
                    [
                        "**Classmate:** Alright. Just checking.",
                        "**Hati:** You stayed on your feet and kept a boundary. That's valid too.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "anger":
            if ph == 0:
                if "sorry" in ut or "stressed" in ut:
                    data["phys_r_ph"] = 1
                    return self._payload(
                        [
                            "**Classmate:** No worries. Hope you feel better.",
                            "**Hati:** You repaired the moment. That's a skill.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                data["phys_r_ph"] = 1
                return self._payload(
                    [
                        "**Narrator:** Classmate walks away, a little offended.",
                        "**Hati:** That was rough—but you can reset next time with a softer first beat.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        if br == "freeze":
            if ph == 0:
                if ut.strip().lower() == "yeah" or "yeah" in ut:
                    data["phys_r_ph"] = 1
                    return self._payload(
                        [
                            "**Classmate:** Okay... well, I have to go now. Nice to see you.",
                            "**Hati:** One word broke the freeze. That's enough for today.",
                        ],
                        {"type": "buttons", "options": ["Continue"]},
                    )
                return self._payload(
                    [
                        "**Narrator:** Classmate looks concerned, pulls out phone.",
                        "**Classmate:** Do you need help?",
                        "**Hati:** You're frozen. It's okay. Let's end the simulation and debrief.",
                    ],
                    {"type": "buttons", "options": ["Continue"]},
                )
            return _to_scene4()

        return _to_scene4()

    def _scene2_line_choice(self, data):
        g = (data.get("goal_text") or "").strip() or "your small win"
        return self._payload(
            [
                f"**Hati:** That's a good goal. Keep that in mind. Now, when you walk toward that seat, the person next to it will probably notice you. They might look up. That's normal. Let's think about what you could say—just a simple line, nothing fancy.",
                "**Hati:** Here are some options, or you can create your own:",
            ],
            {
                "type": "buttons",
                "options": [
                    "Is this seat taken?",
                    "Hey, is anyone sitting here?",
                    "Just a nod and smile, no words",
                    "I'll type my own line (custom)",
                ],
            },
        )

    def _scene2_ready(self, data):
        return self._payload(
            [
                "**Hati:** Good. You have your line. Now let's take a breath together before you go.",
                "**Hati:** Remember: you're not performing. You're just a person existing in a space, like everyone else. Ready?",
            ],
            {"type": "buttons", "options": ["Approach Seat"]},
        )

    def _scene3_npc_prompt(self, data):
        nm = (data.get("user_name") or "there").strip() or "there"
        return self._payload(
            [
                "**Narrator:** You approach the empty seat. The student looks up from their phone.",
                f"**Student:** Oh, hey. Did you need this seat?",
                "**Hati:** Say or type what you do next—your prepared line or your own words. For example you might ask if the seat is free, introduce yourself as "
                f"{nm}, nod and sit, or something else short.",
            ],
            {"type": "text_input", "placeholder": "What you say or do next..."},
        )

    def _scene3_npc_reaction(self, data, branch):
        sk = data.get("scenario_key", "")
        nm = (data.get("user_name") or "there").strip() or "there"
        if branch == "risk":
            messages = [
                f"**Student:** Yeah, go for it. I'm Skye, by the way. This class seems intense, huh?",
                "**Hati:** Nice. They responded warmly. Notice how that feels in your body right now compared to before. This is good practice—you just did it.",
            ]
        elif branch == "nod":
            messages = [
                "**Narrator:** Student nods back and returns to their phone.",
                "**Hati:** That's okay. You did the minimum and it was fine. No rejection, no drama. You're in the seat. That's a win. If you want, you could say something later—but no pressure.",
            ]
        elif branch == "high_anxiety" and sk == "fsn_classroom":
            return self._payload(
                [
                    '**Student:** "Oh... okay. No problem."',
                    "**Hati:** Hey, that felt awkward, I know. But look—nothing bad happened. They weren't mean. You're still standing here, and we can try again. Let's take a breath. What do you need right now?",
                ],
                {
                    "type": "buttons",
                    "options": [
                        "Try again with a simple line",
                        "Debrief with Hati now",
                    ],
                },
            )
        else:
            messages = [
                '**Student:** "Oh... okay. No problem."',
                "**Hati:** Awkward, but not catastrophic. They weren't mean. Breathe.",
            ]
        return self._payload(messages, {"type": "buttons", "options": ["Continue"]})

    def _scene4_debrief_intro(self, data):
        theme = data.get("theme", "")
        sk = data.get("scenario_key", "")
        if theme == "Fear of Strangers & New People" and sk == "fsn_classroom":
            lead = "**Hati:** Okay. Let's pause and reflect. You just did something that takes courage—you approached a stranger."
            messages = [
                lead,
                "**Hati:** Let's compare what you predicted would happen with what actually happened.",
            ]
            return self._payload(messages, {"type": "buttons", "options": ["Continue"]})
        lead = {
            "Fear of Authority": "**Hati:** Let's pause and reflect. You just did something that takes real courage, approaching an authority figure with a request. No matter how it went, you showed up.",
            "Fear of Strangers & New People": "**Hati:** Let's pause and reflect. You just did something that feels scary to many people, approaching a stranger in public. No matter how it went, you showed up.",
            "Fear of Being Observed & Performing": "**Hati:** Let's pause and reflect. You just stood in front of five professors and presented your work. That's objectively difficult. No matter how it went, you faced the fear.",
            "Fear of Social Gatherings": "**Hati:** Let's pause and reflect. You walked into a party with unfamiliar people. That alone takes courage.",
            "Fear of Negative Evaluation & Embarrassment": "**Hati:** Let's pause and reflect. You faced criticism in a group setting. That's one of the hardest social situations.",
            "Physiological Symptoms": "**Hati:** Let's pause and reflect. You experienced visible physical symptoms in public, and someone noticed. That's a fear many people with anxiety have.",
        }.get(theme, "**Hati:** Let's pause and reflect. You showed up for something socially demanding.")
        messages = [
            lead,
            "**Hati:** Let's compare what you predicted with what actually happened.",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Continue"]})

    def _scene4_predicted(self, data):
        theme = data.get("theme", "")
        sk = data.get("scenario_key", "")

        if theme == "Fear of Strangers & New People" and sk == "fsn_classroom":
            q = "**Hati:** On a scale of 0-10, how anxious did you **expect** to feel during that interaction?"
        elif theme == "Fear of Negative Evaluation & Embarrassment":
            q = "**Hati:** On a scale of 0-10, how anxious did you **expect** to feel before sharing?"
        elif theme == "Physiological Symptoms":
            q = "**Hati:** On a scale of 0-10, how anxious did you **expect** to feel when the classmate approached?"
        elif theme == "Fear of Social Gatherings":
            q = "**Hati:** On a scale of 0-10, how anxious did you **expect** to feel before the interaction?"
        else:
            q = "**Hati:** On a scale of 0-10, how anxious did you **expect** to feel?"
        return self._payload(
            [q],
            {
                "type": "buttons",
                "options": [
                    "0", "1", "2", "3", "4",
                    "5", "6", "7", "8", "9", "10"
                ],
            },
        )

    def _scene4_actual(self, data):
        theme = data.get("theme", "")
        sk = data.get("scenario_key", "")
        if theme == "Fear of Strangers & New People" and sk == "fsn_classroom":
            q = "**Hati:** On a scale of 0-10, how anxious did you **actually** feel?"
        elif theme == "Fear of Negative Evaluation & Embarrassment":
            q = "**Hati:** On a scale of 0-10, how anxious did you **actually** feel during the criticism?"
        elif theme == "Physiological Symptoms":
            q = "**Hati:** On a scale of 0-10, how anxious did you **actually** feel during the interaction?"
        elif theme == "Fear of Social Gatherings":
            q = "**Hati:** On a scale of 0-10, how anxious did you **actually** feel during the interaction?"
        else:
            q = "**Hati:** On a scale of 0-10, how anxious did you **actually** feel?"
        return self._payload(
            [q],
            {
                "type": "buttons",
                "options": [
                    "0", "1", "2", "3", "4",
                    "5", "6", "7", "8", "9", "10"
                ],
            },
        )

    def _scene4_fsg_cause(self):
        return self._payload(
            [
                "**Hati:** What do you think caused the difference?",
            ],
            {"type": "text_input", "placeholder": "Optional — type your thoughts, or type skip"},
        )

    def _scene4_fsg_badness(self):
        return self._payload(
            ["**Hati:** Was that as bad as you feared? Rate it 0-10."],
            {
                "type": "buttons",
                "options": [
                    "0", "1", "2", "3", "4",
                    "5", "6", "7", "8", "9", "10"
                ],
            },
        )

    def _scene4_fsg_goal_done_view(self, data):
        g = data.get("goal_text") or "[your goal]"
        return self._payload(
            [
                f"**Hati:** You set a goal: {g}. Did you achieve it?",
                "**Hati:** That's the most important metric. If you achieved your goal, you succeeded. Everything else is extra. If not, that is still fine. You earn a score by trying.",
            ],
            {"type": "buttons", "options": ["Yes", "Partially", "No"]},
        )

    def _scene4_fne_observe(self):
        return self._payload(
            [
                "**Hati:** Did Carlo's criticism actually hurt you physically or end your academic career? Or was it just uncomfortable?",
            ],
            {"type": "text_input", "placeholder": "Type your response..."},
        )

    def _scene4_fne_severity(self):
        return self._payload(
            ["**Hati:** Rate how bad the actual outcome was (0 = nothing, 10 = disaster)."],
            {"type": "text_input", "placeholder": "Type a number 0-10..."},
        )

    def _scene4_fne_goal_done_view(self, data):
        g = data.get("goal_text") or "[your goal]"
        return self._payload(
            [f"**Hati:** You set a goal: {g}. Did you achieve it?"],
            {"type": "buttons", "options": ["Yes", "Partially", "No"]},
        )

    def _scene4_bad(self):
        return self._payload(
            ["**Hati:** Did anything bad actually happen?"],
            {"type": "buttons", "options": ["Yes", "No", "Not sure"]},
        )

    def _scene4_bad_detail(self):
        return self._payload(
            ["**Hati:** What happened?"],
            {"type": "text_input", "placeholder": "Optional — describe what happened"},
        )

    def _scene4_credit(self):
        return self._payload(
            ["**Hati:** What's one small thing you can give yourself credit for?"],
            {"type": "text_input", "placeholder": "Type your response..."},
        )

    def _scene4_class_npc_reflect_view(self):
        return self._payload(
            [
                "**Hati:** Earlier, you were worried about how the other person would respond. Now that it's over, what do you notice about them? Were they scary? Friendly? Neutral? Most people are just focused on themselves, not judging you.",
            ],
            {"type": "text_input", "placeholder": "Optional — type what you noticed, or type skip"},
        )

    def _scene4_personalized(self, data):
        theme = data.get("theme", "")
        pred = data.get("predicted_anxiety")
        actual = data.get("actual_anxiety")
        br = data.get("story_branch", "")
        bad = (data.get("bad_happened") or "").lower()

        if theme == "Fear of Negative Evaluation & Embarrassment":
            style_msgs = {
                "calm": (
                    "**Hati:** You handled criticism with poise. You didn't collapse or attack. "
                    "That's a skill that will serve you in jobs, relationships, and life."
                ),
                "curious": (
                    "**Hati:** You turned a potential conflict into collaboration. "
                    "That's advanced emotional intelligence. Most people can't do that."
                ),
                "apologetic": (
                    "**Hati:** You defaulted to apology. That's common when we fear negative evaluation. "
                    "But apologizing for being human gives others permission to criticize more. "
                    "Next time, try 'I hear you, and I'll improve it.'"
                ),
                "anger": (
                    "**Hati:** Anger is often fear in disguise. If you felt scared of looking stupid, that's okay. "
                    "But anger pushes people away. Next time, take a breath and say 'Let me explain.'"
                ),
                "freeze": (
                    "**Hati:** Freezing is a normal fear response. But silence can be interpreted as agreement or weakness. "
                    "Next time, try a simple 'I'll get back to you' – it buys time without collapsing."
                ),
            }
            msg = style_msgs.get(br, style_msgs["calm"])
        elif theme == "Fear of Strangers & New People" and data.get("scenario_key") == "fsn_classroom":
            if pred is None or actual is None:
                msg = "**Hati:** Thank you for being honest. You showed up and practiced."
            elif actual < pred:
                msg = (
                    "**Hati:** Look at that—your actual anxiety was lower than you predicted. That's evidence. "
                    "**Hati:** Your brain predicted danger, but reality was safer or more neutral. Every time this happens, "
                )
            elif actual == pred:
                msg = (
                    "**Hati:** Thank you for being honest. It's still hard, and that's okay. What matters is that you did it anyway. "
                    "**Hati:** That's courage—feeling the fear and doing it. We'll keep practicing."
                )
            else:
                msg = (
                    "**Hati:** I appreciate your honesty. Today was harder than expected, and that happens. "
                    "**Hati:** The fact that you stayed and tried anyway—that's resilience. Let's try an even bigger step next time."
                )
        elif pred is None or actual is None:
            msg = "**Hati:** Thank you for being honest. You showed up and practiced."
        elif br == "freeze" or "avoid" in br or (bad.startswith("y") and "left" in (data.get("bad_detail") or "").lower()):
            msg = (
                "**Hati:** You chose to step back—that's information, not failure. "
                "**Hati:** Next time we can try a smaller step (a single word, standing nearby, or ten seconds in the room)."
            )
        elif actual < pred:
            msg = (
                "**Hati:** Look at that—your actual anxiety was lower than predicted. "
                "**Hati:** Your brain predicted danger, but reality was manageable. That's evidence you can use next time."
            )
        elif actual == pred:
            msg = (
                "**Hati:** It was as hard as you thought, but you did it anyway. That's courage. Each time, it can get a little easier."
            )
        else:
            msg = (
                "**Hati:** Today was extra hard. That happens. The fact that you tried—or even considered trying—is a step."
            )
        return self._payload([msg], {"type": "buttons", "options": ["Continue"]})

    def _scene4_reflection(self, data):
        theme = data.get("theme", "")
        if theme == "Fear of Negative Evaluation & Embarrassment":
            messages = [
                "**Hati:** What feels different now that the moment has passed?",
            ]
        elif theme == "Physiological Symptoms":
            messages = [
                "**Hati:** Did the classmate react with disgust—or mostly concern or neutrality?",
            ]
        else:
            messages = [
                "**Hati:** What do you notice now about how others showed up—scary, neutral, kinder than expected?",
            ]
        return self._payload(messages, {"type": "text_input", "placeholder": "Type your response..."})

    def _scene5_coping(self, data):
        emotion = (data.get("emotion") or "").lower()
        theme = data.get("theme", "")
        sk = data.get("scenario_key", "")
        tool = None
        if theme == "Fear of Authority":
            if emotion in ["anger", "angry", "irritable", "disgust"]:
                tool = "**Hati:** Pause and label: I notice I'm feeling angry because I'm scared. Naming it reduces its power."
            elif emotion in ["sad"]:
                tool = "**Hati:** Smallest possible action: say one word—'Professor.' That's enough to start."
            elif emotion in ["neutral"]:
                tool = "**Hati:** Anchor: remember a time you handled a difficult conversation—you've done hard things before."
            else:
                tool = "**Hati:** Grounding before speaking: press feet into the floor, feel your back, rehearse your first sentence silently once."
        elif theme == "Fear of Strangers & New People" and sk == "fsn_classroom":
            pies = (data.get("pies_emotional") or "").lower()
            emo_src = pies or emotion
            if any(x in emo_src for x in ["irritable", "anger", "angry", "disgust"]):
                tool = (
                    "**Hati:** The 5-4-3-2-1 grounding: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste."
                )
            elif "sad" in emo_src:
                tool = (
                    "**Hati:** Opposite action: Even if you don't feel like it, give one small smile to the person next to you. "
                    "**Hati:** Action can change feeling."
                )
            elif "neutral" in emo_src:
                tool = (
                    "**Hati:** Body scan: Just notice your feet on the floor, your back against the chair. You're here. You're okay."
                )
            elif any(x in emo_src for x in ["happy", "relief", "calm", "surprise"]):
                tool = "**Hati:** Savoring: Stay with this good feeling for 10 seconds. Let it sink in. You earned it."
            else:
                tool = "**Hati:** Box breathing: Breathe in for 4, hold for 4, out for 4, hold for 4. Try it once now."
        elif theme == "Fear of Strangers & New People":
            if emotion in ["anger", "angry", "disgust"]:
                tool = "**Hati:** The 'just business' reframe: I'm not asking for friendship—I'm claiming a seat. Transaction, not test."
            elif emotion in ["sad"]:
                tool = "**Hati:** One micro-action: point at the seat. No words—the stranger will usually understand."
            elif emotion in ["neutral"]:
                tool = "**Hati:** Anchor to routine: same script as ordering coffee—simple script, simple response."
            elif emotion in ["happy", "surprise"]:
                tool = "**Hati:** Savor and repeat: you did well—use the same short script next time."
            else:
                tool = "**Hati:** The 5-second rule: count down 5-4-3-2-1, then move before your brain talks you out of it."
        elif theme == "Fear of Being Observed & Performing":
            if emotion in ["anger", "angry", "disgust"]:
                tool = "**Hati:** Pause and label: I'm angry because I'm scared of looking unprepared. Naming lowers intensity."
            elif emotion in ["sad"]:
                tool = "**Hati:** Micro-commitment: say only the first sentence—then you can stop. Starting is often the hardest part."
            elif emotion in ["neutral"]:
                tool = "**Hati:** Anchor: recall a time you succeeded in a performance—you still have that capability."
            elif emotion in ["happy", "calm"]:
                tool = "**Hati:** Savor: notice pace and breath—repeat that recipe next time."
            else:
                tool = "**Hati:** Spotlight reframe: they're evaluating your work, not your worth as a person. Or try 5-4-3-2-1 grounding now."
        elif theme == "Fear of Social Gatherings":
            if emotion in ["anger", "angry", "disgust"]:
                tool = "**Hati:** Soft start: 'Hey, I'm a bit nervous, but wanted to say hi.' Vulnerability often invites warmth."
            elif data.get("story_branch") == "avoid":
                tool = "**Hati:** Next time aim for one tiny exchange—even 'Hi, I like your shirt.' Rehearse that one line."
            elif emotion in ["sad"]:
                tool = "**Hati:** Observer reframe: I'm gathering data on how people interact—write down three neutral observations."
            elif emotion in ["fear", "anxious"]:
                tool = "**Hati:** Curiosity shift: ask 'How do you know the host?' instead of only monitoring yourself."
            else:
                tool = "**Hati:** Replay the moment you felt most included—hold that feeling ten seconds (savoring)."
        elif theme == "Fear of Negative Evaluation & Embarrassment":
            br = data.get("story_branch", "")
            if br == "anger":
                tool = "**Hati:** Thanks-for-feedback pivot: 'Thanks, I'll consider that' buys time and lowers heat. Rehearse it neutrally."
            elif br == "freeze":
                tool = "**Hati:** Broken-record safety line: 'Let me think about that.' or 'I'll send more by email.' Memorize one."
            elif br == "apologetic":
                tool = "**Hati:** Pause and breathe once before replying so you choose your words instead of reflex apologizing."
            elif br == "curious":
                tool = "**Hati:** Replay the moment you stayed collaborative—that calm is a resource. Savor ten seconds."
            else:
                tool = "**Hati:** Boundary line practice: 'That's what I have ready; I'll add detail by the deadline.'"
        elif theme == "Physiological Symptoms":
            phys = (data.get("pies_physical") or "").lower()
            if emotion in ["fear", "anxious"]:
                tool = (
                    "**Hati:** So-what reframe: if they notice, worst case they think I'm tired or warm. "
                    "**Hati:** Say 'so what' three times softly."
                )
            elif "shake" in phys:
                tool = "**Hati:** Pocket trick: hands in pockets or clasped—reduces visible shake and gives your body a job."
            elif "flush" in phys:
                tool = "**Hati:** Temperature pivot: 'Warm out here, isn't it?'—rehearse once."
            else:
                tool = "**Hati:** Box breathing: in 4, hold 4, out 4, hold 4—one full cycle now."
        else:
            if emotion in ["anxious", "fear"]:
                tool = "**Hati:** Box breathing: in 4, hold 4, out 4, hold 4."
            elif emotion in ["anger", "angry", "disgust"]:
                tool = "**Hati:** 5-4-3-2-1 grounding."
            elif emotion in ["sad"]:
                tool = "**Hati:** Opposite action: one small nod or smile."
            else:
                tool = "**Hati:** Feet on the floor, shoulders loose, one slow exhale."
        data["last_coping_tool"] = tool
        intro = "**Hati:** Based on what you experienced, here's a tool you can use next time:"
        if sk == "fsn_classroom":
            intro = "**Hati:** Based on what you just experienced, I want to give you a small tool to carry with you."
        messages = [
            intro,
            tool,
            "**Hati:** Want to try this together now?",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Yes", "Maybe later", "No"]})

    def _scene6_closing(self, data):
        theme = data.get("theme", "")
        sk = data.get("scenario_key", "")
        br = data.get("story_branch", "")
        tool_rem = (data.get("last_coping_tool") or "").strip()
        insight = "**Hati:** You practiced showing up—that builds evidence for your future self."
        if sk == "fsn_classroom":
            insight = (
                "**Hati:** You rehearsed a first-day moment many students avoid. The goal wasn't perfection—it was showing up for yourself."
            )
            extra_tool = f" Your practice tool from earlier: {tool_rem}" if tool_rem else ""
            messages = [
                "**Hati:** Here's what I want you to remember from today:",
                insight + extra_tool,
                "**Hati:** You showed up. You practiced. You're building skills, one moment at a time. I'm proud of you. And I'll be here next time you need to practice.",
                "**Hati:** For now, take a deep breath. You've earned a moment of rest.",
            ]
            return self._payload(messages, {"type": "buttons", "options": ["Finish"]})
        if theme == "Fear of Authority":
            insight = "**Hati:** Authority figures are people with more experience in one area—they were students too. You can make respectful requests without being a burden."
        elif theme == "Fear of Strangers & New People":
            insight = "**Hati:** Most strangers are neutral and busy. Asking for a seat is ordinary—not a performance."
        elif theme == "Fear of Being Observed & Performing":
            insight = "**Hati:** Every panelist once stood where you stood. They know how hard this is; most want you to succeed."
        elif theme == "Fear of Social Gatherings":
            insight = "**Hati:** Many guests feel a little awkward too—they're just better at hiding it. Micro-steps still count as courage."
        elif theme == "Fear of Negative Evaluation & Embarrassment":
            insight = "**Hati:** One person's tone doesn't define your worth. Feedback can be information, not a verdict."
        elif theme == "Physiological Symptoms":
            insight = "**Hati:** Your body can spike adrenaline without danger. Symptoms are uncomfortable, not proof of catastrophe."
        if br == "anxious":
            insight += "**Hati:** You stayed in the hard version—and that still matters."
        elif br in ("anger",):
            insight += "**Hati:** Repair and reset are skills too—you can practice a softer start next time."
        elif br == "curious":
            insight += "**Hati:** Small talk is a skill—you practiced turning nerves into connection."
        elif br in ("freeze", "avoid"):
            insight += "**Hati:** If you backed away, we'll shrink the next practice until it fits—progress isn't only loud moments."
        messages = [
            "**Hati:** Here's what I want you to remember:",
            insight,
            "**Hati:** You showed up today. That's a win. I'll be here next time.",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Finish"]})

    def _scene7_dashboard(self):
        messages = [
            "**Hati:** Great work today. I've logged your emotions and progress.",
            "**Hati:** Over time, you'll start to see patterns, what triggers your anxiety, what helps, and how far you've come.",
            "**Hati:** Want to see your progress so far?",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Open Progress", "Close"]})

    def _complete(self):
        return self._payload(
            ["CONGRATULATIONS! YOU COMPLETED A SCENARIO!"],
            {"type": "text_input"},
        )

    def _complete_and_dashboard(self, data=None):
        data = data or {}
        sk = data.get("scenario_key", "")
        congrats = "CONGRATULATIONS! YOU COMPLETED A SCENARIO!"
        if sk == "fsn_classroom":
            congrats = "CONGRATULATIONS! YOU JUST COMPLETED A SCENARIO!"
        hati_extra = "**Hati:** Great work. I've logged your emotions. Over time you'll see patterns—what feels hardest and how confidence grows."
        if sk == "fsn_classroom":
            hati_extra = (
                "**Hati:** Great work today. I've logged your emotions and progress. Over time, you'll start to see patterns—"
                "what triggers your anxiety, what helps, and how far you've come."
            )
        messages = [
            congrats,
            hati_extra,
            "Want to see your progress so far?",
        ]
        return self._payload(messages, {"type": "buttons", "options": ["Open Progress", "Close"]})

    def _end(self):
        return self._payload([], {"type": "text_input"})


    def _npc_branch_seat(self, response_text, emotion):
        text = (response_text or "").lower()
        emo = self._detected_emotion_lower(emotion)
        if "nod" in text or "smile" in text or "just nods" in text or "while sitting" in text:
            return "nod"
        if any(k in text for k in ["sorry", "never mind", "nevermind"]) or (
            "uh" in text and "free" not in text and "thanks" not in text
        ):
            return "high_anxiety"
        if "is it free" in text or "simple, practical" in text:
            return "risk"
        if "by the way" in text or "introduces self" in text:
            return "risk"
        if emo in ("sad", "anxious", "fear") and len(text) < 100:
            if "free" in text or ("thanks" in text and "yeah" in text):
                return "risk"
            return "high_anxiety"
        return "risk"

    def _npc_branch(self, response_text, emotion):
        return self._npc_branch_seat(response_text, emotion)

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


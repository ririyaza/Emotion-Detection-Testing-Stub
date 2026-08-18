import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import EmotionDatabase
from scenario_engine import ScenarioEngine


class ScenarioEngineFearOfAuthorityTest(unittest.TestCase):
    def test_fear_of_authority_flow_reaches_debrief(self):
        engine = ScenarioEngine(storage_path='__tmp_test_sessions.json')
        session_id, payload = engine.start_session(theme='Fear of Authority', scenario_key='foa_supervisor')

        self.assertEqual(engine.sessions[session_id]['step'], 'scene0_greet')

        response = engine.handle_step(session_id, {'text': 'Begin', 'emotion': 'anxious'})
        self.assertTrue(
            any('department office' in message.lower() or 'physical' in message.lower() for message in response.get('messages', []))
        )

        engine.handle_step(session_id, {'text': 'Tense', 'emotion': 'anxious'})
        engine.handle_step(session_id, {'text': 'Anxious', 'emotion': 'anxious'})
        engine.handle_step(session_id, {'text': 'The professor\'s stern face', 'emotion': 'anxious'})

        state = engine.sessions[session_id]
        self.assertEqual(state['step'], 'foa_s2_script')


class FirebaseAuthFailureTest(unittest.TestCase):
    def test_invalid_jwt_disables_firebase_client(self):
        class FakeQuery:
            def where(self, *args, **kwargs):
                raise Exception("('invalid_grant: Invalid JWT Signature.', {'error': 'invalid_grant', 'error_description': 'Invalid JWT Signature.'})")

        class FakeClient:
            def collection(self, *args, **kwargs):
                return FakeQuery()

        db = EmotionDatabase.__new__(EmotionDatabase)
        db.initialized = True
        db.client = FakeClient()
        db.credentials_path = "firebase/not-real.json"
        db.project_id = "test-project"

        result = db.get_unfinished_scenarios("user-123")

        self.assertIsNone(result)
        self.assertFalse(db.initialized)
        self.assertIsNone(db.client)


if __name__ == '__main__':
    unittest.main()

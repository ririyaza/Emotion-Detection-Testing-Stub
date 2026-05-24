import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os
from pathlib import Path

class EmotionDatabase:
    """Firebase Firestore handler for storing emotion logs."""
    
    def __init__(self, credentials_path=None, project_id=None):
        """
        Initialize Firebase connection.
        
        Args:
            credentials_path: Path to Firebase service account JSON file
            project_id: Firebase project ID
        """
        self.credentials_path = credentials_path
        self.project_id = project_id or os.getenv("FIREBASE_PROJECT_ID", "hati-25259")
        self.initialized = False
        self.client = None
        
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase app with credentials."""
        try:
            if self.credentials_path and Path(self.credentials_path).exists():
                cred = credentials.Certificate(self.credentials_path)
                firebase_admin.initialize_app(cred)
                self.client = firestore.client()
                self.initialized = True
            else:
                print(f"Warning: Firebase credentials not found at {self.credentials_path}")
                print("Emotion logging will be disabled. Set up credentials to enable.")
        except Exception as e:
            print(f"Firebase initialization error: {e}")
            print("Emotion logging will be disabled.")
    
    def log_emotion(self, user_id, scenario_id, emotion_data):
        """
        Log emotion data to Firestore under users/{userId}/scenarios/{scenarioId}/emotionLogs.
        
        Args:
            user_id: Unique user identifier
            scenario_id: Unique scenario/session identifier
            emotion_data: Dictionary containing emotion information
                - emotion: The detected or provided emotion
                - step: Current scenario step
                - theme: Scenario theme (e.g., "Fear of Authority")
                - scenario_key: Scenario key identifier (e.g., "foa_supervisor")
                - user_response: User's text input (limited to 500 chars)
                - story_branch: Story branch taken (if applicable)
                - confidence: Confidence score (0-1, if applicable)
                - timestamp: Will be auto-added (ISO format)
                - audio_detected: Whether emotion was from audio (optional)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.client is None:
            return False
        
        try:
            if 'timestamp' not in emotion_data:
                emotion_data['timestamp'] = datetime.now().isoformat()
            
            log_ref = self.client.collection('users')
            log_ref = log_ref.document(user_id)
            log_ref = log_ref.collection('scenarios')
            log_ref = log_ref.document(scenario_id)
            log_ref = log_ref.collection('emotionLogs')
            log_ref.add(emotion_data)
            
            return True
        except Exception as e:
            print(f"Error logging emotion to Firebase: {e}")
            return False
    
    def get_emotion_logs(self, user_id, scenario_id):
        """
        Retrieve all emotion logs for a user's scenario.
        
        Args:
            user_id: User identifier
            scenario_id: Scenario identifier
        
        Returns:
            dict: Dictionary of emotion logs or None if error
        """
        if not self.initialized or self.client is None:
            return None
        
        try:
            logs_collection = self.client.collection('users')
            logs_collection = logs_collection.document(user_id)
            logs_collection = logs_collection.collection('scenarios')
            logs_collection = logs_collection.document(scenario_id)
            logs_collection = logs_collection.collection('emotionLogs')
            docs = logs_collection.stream()
            return {doc.id: doc.to_dict() for doc in docs}
        except Exception as e:
            print(f"Error retrieving emotion logs: {e}")
            return None
    
    def get_user_scenarios(self, user_id):
        """
        Get all scenarios for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            dict: Dictionary of scenarios or None if error
        """
        if not self.initialized or self.client is None:
            return None
        
        try:
            scenarios_collection = self.client.collection('users')
            scenarios_collection = scenarios_collection.document(user_id)
            scenarios_collection = scenarios_collection.collection('scenarios')
            docs = scenarios_collection.stream()
            return {doc.id: doc.to_dict() for doc in docs}
        except Exception as e:
            print(f"Error retrieving user scenarios: {e}")
            return None

    def create_scenario(self, user_id, scenario_id, metadata):
        """
        Create or update a scenario metadata node for a user.

        Args:
            user_id: User identifier
            scenario_id: Scenario/session identifier
            metadata: Dictionary of metadata to store

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.client is None:
            return False

        try:
            scenario_ref = self.client.collection('users')
            scenario_ref = scenario_ref.document(user_id)
            scenario_ref = scenario_ref.collection('scenarios')
            scenario_ref = scenario_ref.document(scenario_id)
            scenario_ref.set(metadata)
            return True
        except Exception as e:
            print(f"Error creating scenario metadata: {e}")
            return False

    def update_scenario_step(self, user_id, scenario_id, current_step):
        """
        Update the current step of a scenario in Firebase.
        
        Args:
            user_id: User identifier
            scenario_id: Scenario identifier
            current_step: Current scenario step name
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.client is None:
            return False
        
        try:
            scenario_ref = self.client.collection('users')
            scenario_ref = scenario_ref.document(user_id)
            scenario_ref = scenario_ref.collection('scenarios')
            scenario_ref = scenario_ref.document(scenario_id)
            scenario_ref.update({
                'current_step': current_step,
                'updated_at': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            print(f"Error updating scenario step: {e}")
            return False

    def update_scenario_state(self, user_id, scenario_id, state):
        """
        Update the full scenario session state in Firebase.
        
        Args:
            user_id: User identifier
            scenario_id: Scenario identifier
            state: Dictionary containing scenario session state
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.client is None:
            return False

        try:
            scenario_ref = self.client.collection('users')
            scenario_ref = scenario_ref.document(user_id)
            scenario_ref = scenario_ref.collection('scenarios')
            scenario_ref = scenario_ref.document(scenario_id)
            scenario_ref.update({
                'current_step': state.get('step', ''),
                'session_state': state,
                'updated_at': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            print(f"Error updating scenario state: {e}")
            return False

    def get_scenario_state(self, user_id, scenario_id):
        """
        Retrieve the stored session state for a scenario.

        Args:
            user_id: User identifier
            scenario_id: Scenario identifier

        Returns:
            dict or None: Stored session state or None if unavailable
        """
        if not self.initialized or self.client is None:
            return None

        try:
            scenario_ref = self.client.collection('users')
            scenario_ref = scenario_ref.document(user_id)
            scenario_ref = scenario_ref.collection('scenarios')
            scenario_ref = scenario_ref.document(scenario_id)
            doc = scenario_ref.get()
            if doc.exists:
                data = doc.to_dict()
                return data.get('session_state')
            return None
        except Exception as e:
            print(f"Error retrieving scenario state: {e}")
            return None

    def get_unfinished_scenarios(self, user_id):
        """
        Get all unfinished scenarios for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            dict: Dictionary of unfinished scenarios {scenario_id: metadata} or None if error
        """
        if not self.initialized or self.client is None:
            return None
        
        try:
            scenarios_collection = self.client.collection('users')
            scenarios_collection = scenarios_collection.document(user_id)
            scenarios_collection = scenarios_collection.collection('scenarios')
            query = scenarios_collection.where('current_step', '!=', 'complete')
            docs = query.stream()
            return {doc.id: doc.to_dict() for doc in docs}
        except Exception as e:
            print(f"Error retrieving unfinished scenarios: {e}")
            return None
    
    def delete_emotion_log(self, user_id, scenario_id, log_id):
        """
        Delete a specific emotion log entry.
        
        Args:
            user_id: User identifier
            scenario_id: Scenario identifier
            log_id: Emotion log ID (Firebase key)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.initialized or self.client is None:
            return False
        
        try:
            log_ref = self.client.collection('users')
            log_ref = log_ref.document(user_id)
            log_ref = log_ref.collection('scenarios')
            log_ref = log_ref.document(scenario_id)
            log_ref = log_ref.collection('emotionLogs')
            log_ref.document(log_id).delete()
            return True
        except Exception as e:
            print(f"Error deleting emotion log: {e}")
            return False

emotion_db = None

def init_emotion_database(credentials_path=None, project_id=None):
    """Initialize global emotion database instance."""
    global emotion_db
    emotion_db = EmotionDatabase(credentials_path, project_id)
    return emotion_db

def get_emotion_database():
    """Get global emotion database instance."""
    global emotion_db
    if emotion_db is None:
        emotion_db = EmotionDatabase()
    return emotion_db

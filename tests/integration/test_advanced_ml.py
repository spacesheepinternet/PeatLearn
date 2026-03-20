#!/usr/bin/env python3
"""
Comprehensive Test Suite for PeatLearn Advanced ML Features
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

async def test_complete_pipeline():
    """Test the complete ML pipeline."""
    
    print("🧪 Running Complete Advanced ML Pipeline Test...")
    
    try:
        # Import all components
        from peatlearn.personalization.neural import (
            AdvancedPersonalizationEngine,
            UserInteraction,
            LearningState
        )
        from peatlearn.personalization.rl_agent import (
            AdaptiveLearningAgent,
            LearningEnvironmentState
        )
        from peatlearn.personalization.knowledge_graph import AdvancedKnowledgeGraph
        
        # Initialize components
        print("Initializing components...")
        
        personalization = AdvancedPersonalizationEngine()
        await personalization.initialize_models(num_users=50, num_content=100, num_topics=10)
        
        rl_agent = AdaptiveLearningAgent(state_dim=32, content_action_dim=20)
        
        knowledge_graph = AdvancedKnowledgeGraph()
        await knowledge_graph.initialize_models()
        
        print("✅ All components initialized")
        
        # Test personalization
        print("Testing personalization...")
        recommendations = await personalization.get_content_recommendations("test_user", 5)
        assert len(recommendations) == 5, "Should return 5 recommendations"
        print(f"✅ Got {len(recommendations)} recommendations")
        
        # Test RL agent
        print("Testing RL agent...")
        test_state = LearningEnvironmentState(
            user_id="test_user",
            current_topic_mastery={"thyroid": 0.6, "metabolism": 0.4},
            recent_performance=[0.7, 0.8, 0.6],
            time_in_session=15.0,
            difficulty_progression=[0.5, 0.6, 0.7],
            engagement_level=0.8,
            fatigue_level=0.2,
            topics_covered_today=2,
            consecutive_correct=3,
            consecutive_incorrect=1,
            preferred_learning_style="analytical",
            context_features=np.zeros(20)
        )
        
        action, metadata = await rl_agent.select_content_action(test_state)
        assert isinstance(action, int), "Action should be an integer"
        print(f"✅ RL agent selected action: {action}")
        
        # Test knowledge graph
        print("Testing knowledge graph...")
        test_concepts = await knowledge_graph.extract_concepts_from_text(
            "Progesterone helps reduce stress and improves thyroid function. "
            "Estrogen can cause inflammation and metabolic problems."
        )
        assert len(test_concepts) > 0, "Should extract some concepts"
        print(f"✅ Extracted {len(test_concepts)} concepts")
        
        print("\n🎉 Complete ML Pipeline Test Passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np
    success = asyncio.run(test_complete_pipeline())
    sys.exit(0 if success else 1)
    
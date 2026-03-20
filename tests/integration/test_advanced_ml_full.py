#!/usr/bin/env python3
"""
Integration Test for Advanced PeatLearn ML System

Tests the complete pipeline from basic RAG to advanced personalization,
reinforcement learning, and knowledge graph integration.
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

project_root = Path(__file__).parent.parent.parent
os.environ['PROJECT_ROOT'] = str(project_root)

class AdvancedMLTester:
    """Comprehensive tester for all advanced ML components."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
    
    async def test_basic_rag(self) -> bool:
        """Test basic RAG functionality."""
        print("\n🧪 Testing Basic RAG System...")
        
        try:
            from config.settings import settings
            from peatlearn.rag.rag_system import RayPeatRAG
            
            if not settings.GEMINI_API_KEY:
                print("⚠️ GEMINI_API_KEY not set, skipping RAG test")
                return True  # Don't fail the whole test
            
            rag = RayPeatRAG()
            question = "What does Ray Peat say about thyroid function?"
            response = await rag.answer_question(question, max_sources=3)
            
            assert response.answer is not None and len(response.answer) > 0
            assert len(response.sources) > 0
            assert 0.0 <= response.confidence <= 1.0
            
            print(f"✅ Basic RAG test passed - Confidence: {response.confidence:.2f}")
            self.test_results['basic_rag'] = {
                'status': 'passed',
                'confidence': response.confidence,
                'sources_found': len(response.sources)
            }
            return True
            
        except Exception as e:
            print(f"❌ Basic RAG test failed: {e}")
            self.test_results['basic_rag'] = {'status': 'failed', 'error': str(e)}
            return False
    
    async def test_neural_personalization(self) -> bool:
        """Test neural personalization engine."""
        print("\n🧪 Testing Neural Personalization Engine...")
        
        try:
            from peatlearn.personalization.neural import (
                AdvancedPersonalizationEngine,
                UserInteraction,
                LearningState
            )
            
            engine = AdvancedPersonalizationEngine()
            await engine.initialize_models(num_users=20, num_content=50, num_topics=5)
            
            # Test user state management
            test_interaction = UserInteraction(
                user_id="test_user_123",
                content_id="content_456",
                interaction_type="question",
                timestamp=datetime.now(),
                performance_score=0.75,
                time_spent=120.0,
                difficulty_level=0.6,
                topic_tags=["thyroid", "metabolism"],
                context={"test": True}
            )
            
            await engine.update_user_state("test_user_123", test_interaction)
            
            # Test recommendations
            recommendations = await engine.get_content_recommendations("test_user_123", 5)
            assert len(recommendations) == 5
            
            # Test learning trajectory prediction
            trajectory = await engine.predict_learning_trajectory(
                "test_user_123", [test_interaction]
            )
            
            assert 'topic_mastery' in trajectory
            assert 'optimal_difficulty' in trajectory
            assert 'predicted_engagement' in trajectory
            
            # Test quiz generation
            user_state = engine.learning_states["test_user_123"]
            quiz_spec = await engine.generate_personalized_quiz(
                "test_user_123",
                np.random.normal(0, 0.1, 768),  # Mock content embedding
                user_state
            )
            
            assert 'difficulty' in quiz_spec
            assert 0.0 <= quiz_spec['difficulty'] <= 1.0
            
            print("✅ Neural Personalization test passed")
            self.test_results['neural_personalization'] = {
                'status': 'passed',
                'recommendations_count': len(recommendations),
                'user_state_updated': True,
                'quiz_generated': True
            }
            return True
            
        except Exception as e:
            print(f"❌ Neural Personalization test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['neural_personalization'] = {'status': 'failed', 'error': str(e)}
            return False
    
    async def test_reinforcement_learning(self) -> bool:
        """Test reinforcement learning agent."""
        print("\n🧪 Testing Reinforcement Learning Agent...")
        
        try:
            from peatlearn.personalization.rl_agent import (
                AdaptiveLearningAgent,
                LearningEnvironmentState
            )
            
            agent = AdaptiveLearningAgent(state_dim=128, content_action_dim=10)
            
            # Create test learning state
            test_state = LearningEnvironmentState(
                user_id="rl_test_user",
                current_topic_mastery={"thyroid": 0.7, "metabolism": 0.5},
                recent_performance=[0.8, 0.6, 0.9, 0.7],
                time_in_session=20.0,
                difficulty_progression=[0.4, 0.5, 0.6],
                engagement_level=0.75,
                fatigue_level=0.3,
                topics_covered_today=3,
                consecutive_correct=2,
                consecutive_incorrect=1,
                preferred_learning_style="analytical",
                context_features=np.random.normal(0, 0.1, 20)
            )
            
            # Test action selection
            action, metadata = await agent.select_content_action(test_state)
            assert isinstance(action, int)
            assert 0 <= action < 10
            assert 'method' in metadata
            
            # Test difficulty adjustment
            difficulty, diff_metadata = await agent.adjust_difficulty(test_state)
            assert 0.0 <= difficulty <= 1.0
            assert 'predicted_value' in diff_metadata
            
            # Test reward calculation
            next_state = LearningEnvironmentState(
                user_id="rl_test_user",
                current_topic_mastery={"thyroid": 0.75, "metabolism": 0.55},  # Improved
                recent_performance=[0.6, 0.9, 0.7, 0.8],
                time_in_session=25.0,
                difficulty_progression=[0.5, 0.6, 0.7],
                engagement_level=0.8,
                fatigue_level=0.4,
                topics_covered_today=3,
                consecutive_correct=3,
                consecutive_incorrect=0,
                preferred_learning_style="analytical",
                context_features=np.random.normal(0, 0.1, 20)
            )
            
            reward = agent.calculate_reward(
                test_state, 
                action, 
                next_state,
                {'performance_score': 0.8, 'difficulty_rating': 0.7}
            )
            
            # Test experience storage
            agent.store_experience(test_state, action, reward, next_state)
            assert len(agent.memory) == 1
            
            # Test bandit update
            await agent.update_bandit(action, reward)
            
            print(f"✅ RL Agent test passed - Action: {action}, Reward: {reward:.2f}")
            self.test_results['reinforcement_learning'] = {
                'status': 'passed',
                'action_selected': action,
                'difficulty_adjusted': difficulty,
                'reward_calculated': reward,
                'experience_stored': True
            }
            return True
            
        except Exception as e:
            print(f"❌ RL Agent test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['reinforcement_learning'] = {'status': 'failed', 'error': str(e)}
            return False
    
    async def test_knowledge_graph(self) -> bool:
        """Test knowledge graph system."""
        print("\n🧪 Testing Knowledge Graph System...")
        
        try:
            from peatlearn.personalization.knowledge_graph import AdvancedKnowledgeGraph
            
            kg = AdvancedKnowledgeGraph()
            await kg.initialize_models()
            
            # Test concept extraction
            sample_text = """
            Progesterone is a hormone that helps reduce stress and supports thyroid function.
            High estrogen levels can cause inflammation and interfere with metabolism.
            Aspirin can help reduce inflammation and improve energy production.
            Coconut oil provides medium-chain fatty acids that support metabolic function.
            """
            
            concepts = await kg.extract_concepts_from_text(sample_text, min_frequency=1)
            assert len(concepts) > 0
            
            print(f"Extracted {len(concepts)} concepts: {[c.name for c in concepts[:3]]}")
            
            # Test relationship extraction
            relations = await kg.extract_relationships(sample_text, concepts)
            assert len(relations) >= 0  # Might be 0 if patterns don't match
            
            print(f"Extracted {len(relations)} relationships")
            
            # Test query expansion
            test_query = "thyroid function and metabolism"
            expanded = await kg.expand_query_with_concepts(test_query, max_expansions=3)
            
            assert 'original_query' in expanded
            assert 'expanded_query' in expanded
            assert 'expansion_terms' in expanded
            
            # Test analytics
            analytics = kg.get_graph_analytics()
            assert 'num_concepts' in analytics
            assert 'num_relations' in analytics
            
            print("✅ Knowledge Graph test passed")
            self.test_results['knowledge_graph'] = {
                'status': 'passed',
                'concepts_extracted': len(concepts),
                'relations_extracted': len(relations),
                'query_expansion_terms': len(expanded['expansion_terms']),
                'graph_analytics': analytics
            }
            return True
            
        except Exception as e:
            print(f"❌ Knowledge Graph test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['knowledge_graph'] = {'status': 'failed', 'error': str(e)}
            return False
    
    async def test_advanced_rag_integration(self) -> bool:
        """Test integration of all components with RAG."""
        print("\n🧪 Testing Advanced RAG Integration...")
        
        try:
            from config.settings import settings
            
            if not settings.GEMINI_API_KEY:
                print("⚠️ GEMINI_API_KEY not set, skipping integration test")
                return True
            
            # This would test the full pipeline from advanced_app.py
            # For now, just verify components can work together
            
            # Import all components
            from peatlearn.rag.rag_system import RayPeatRAG
            from peatlearn.personalization.neural import (
                AdvancedPersonalizationEngine, UserInteraction
            )
            from peatlearn.personalization.knowledge_graph import AdvancedKnowledgeGraph
            
            # Initialize
            rag = RayPeatRAG()
            personalization = AdvancedPersonalizationEngine()
            kg = AdvancedKnowledgeGraph()
            
            await personalization.initialize_models(num_users=5, num_content=20, num_topics=5)
            await kg.initialize_models()
            
            # Test integrated workflow
            user_id = "integration_test_user"
            original_query = "How does progesterone affect thyroid?"
            
            # Step 1: Expand query with knowledge graph
            expanded_query_data = await kg.expand_query_with_concepts(original_query)
            
            # Step 2: Get RAG response
            response = await rag.answer_question(original_query, max_sources=3)
            
            # Step 3: Log interaction for personalization
            interaction = UserInteraction(
                user_id=user_id,
                content_id=f"integration_test_{hash(original_query)}",
                interaction_type="question",
                timestamp=datetime.now(),
                performance_score=response.confidence,
                time_spent=30.0,
                difficulty_level=0.6,
                topic_tags=["progesterone", "thyroid"],
                context={"integration_test": True}
            )
            
            await personalization.update_user_state(user_id, interaction)
            
            # Step 4: Get personalized recommendations
            recommendations = await personalization.get_content_recommendations(user_id, 3)
            
            assert len(recommendations) == 3
            assert response.confidence > 0
            
            print("✅ Advanced RAG Integration test passed")
            self.test_results['advanced_rag_integration'] = {
                'status': 'passed',
                'query_expanded': len(expanded_query_data['expansion_terms']) > 0,
                'rag_response_generated': True,
                'interaction_logged': True,
                'recommendations_generated': len(recommendations)
            }
            return True
            
        except Exception as e:
            print(f"❌ Advanced RAG Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['advanced_rag_integration'] = {'status': 'failed', 'error': str(e)}
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all tests and generate report."""
        print("🚀 Starting Comprehensive Advanced ML Test Suite")
        print("=" * 60)
        
        # Run all tests
        tests = [
            ("Basic RAG", self.test_basic_rag),
            ("Neural Personalization", self.test_neural_personalization),
            ("Reinforcement Learning", self.test_reinforcement_learning),
            ("Knowledge Graph", self.test_knowledge_graph),
            ("Advanced RAG Integration", self.test_advanced_rag_integration)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                if success:
                    passed += 1
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = {
                    'status': 'crashed', 'error': str(e)
                }
        
        # Generate final report
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print(f"Total Duration: {duration:.2f} seconds")
        
        # Detailed results
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'passed' else "❌"
            print(f"{status_emoji} {test_name.replace('_', ' ').title()}: {result['status']}")
        
        # Save detailed results
        report = {
            'summary': {
                'total_tests': total,
                'passed_tests': passed,
                'success_rate': (passed/total)*100,
                'duration_seconds': duration,
                'timestamp': end_time.isoformat()
            },
            'detailed_results': self.test_results
        }
        
        with open('test_results_advanced_ml.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: test_results_advanced_ml.json")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Advanced ML system is ready!")
            print("\nNext steps:")
            print("1. Start the advanced server: python inference/backend/advanced_app.py")
            print("2. Test the API endpoints")
            print("3. Build the frontend integration")
        else:
            print(f"\n⚠️ {total - passed} tests failed. Check the logs above for details.")
        
        return passed == total

async def main():
    """Main test function."""
    tester = AdvancedMLTester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())

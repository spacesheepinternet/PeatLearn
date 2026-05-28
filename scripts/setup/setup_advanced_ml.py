#!/usr/bin/env python3
"""
Installation and Setup Script for PeatLearn Advanced ML Components

This script installs all required dependencies and runs initial tests
to ensure the advanced ML/AI features are working correctly.
"""

import subprocess
import sys
import os
from pathlib import Path
import asyncio

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def install_requirements():
    """Install all advanced ML requirements."""
    
    print("📦 Installing Advanced ML Requirements...")
    
    # Basic requirements first
    success = run_command(
        "pip install fastapi uvicorn pydantic aiohttp python-dotenv",
        "Installing basic FastAPI requirements"
    )
    
    if not success:
        return False
    
    # PyTorch (CPU version for compatibility)
    success = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "Installing PyTorch (CPU version)"
    )
    
    if not success:
        return False
    
    # Transformers and NLP libraries
    success = run_command(
        "pip install transformers sentence-transformers datasets tokenizers",
        "Installing Transformers and NLP libraries"
    )
    
    if not success:
        return False
    
    # Graph libraries
    success = run_command(
        "pip install torch-geometric networkx",
        "Installing Graph Neural Network libraries"
    )
    
    if not success:
        return False
    
    # Scientific computing
    success = run_command(
        "pip install numpy pandas scikit-learn scipy matplotlib seaborn",
        "Installing scientific computing libraries"
    )
    
    if not success:
        return False
    
    # Reinforcement Learning
    success = run_command(
        "pip install stable-baselines3 gymnasium",
        "Installing Reinforcement Learning libraries"
    )
    
    if not success:
        return False
    
    # Vector databases and search
    success = run_command(
        "pip install faiss-cpu chromadb",
        "Installing vector database libraries"
    )
    
    if not success:
        return False
    
    # Optimization and experiment tracking
    success = run_command(
        "pip install optuna wandb mlflow",
        "Installing optimization and MLOps libraries"
    )
    
    if not success:
        return False
    
    print("✅ All advanced ML requirements installed successfully!")
    return True

def test_imports():
    """Test that all critical imports work."""
    
    print("\n🧪 Testing Advanced ML Imports...")
    
    import_tests = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("torch_geometric", "PyTorch Geometric"),
        ("networkx", "NetworkX"),
        ("sklearn", "Scikit-learn"),
        ("stable_baselines3", "Stable Baselines 3"),
        ("pinecone", "Pinecone"),
        ("optuna", "Optuna")
    ]
    
    failed_imports = []
    
    for module, name in import_tests:
        try:
            __import__(module)
            print(f"✅ {name} import successful")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n⚠️ Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

async def test_advanced_components():
    """Test the advanced ML components."""
    
    print("\n🔬 Testing Advanced ML Components...")
    
    try:
        # Test Neural Personalization Engine
        print("Testing Neural Personalization Engine...")
        from inference.backend.personalization.neural_personalization import (
            AdvancedPersonalizationEngine
        )
        
        engine = AdvancedPersonalizationEngine()
        await engine.initialize_models(num_users=100, num_content=500, num_topics=20)
        print("✅ Neural Personalization Engine initialized")
        
        # Test RL Agent
        print("Testing Reinforcement Learning Agent...")
        from inference.backend.personalization.rl_agent import AdaptiveLearningAgent
        
        agent = AdaptiveLearningAgent(state_dim=64, content_action_dim=50)
        print("✅ RL Agent initialized")
        
        # Test Knowledge Graph
        print("Testing Knowledge Graph System...")
        from inference.backend.personalization.knowledge_graph import AdvancedKnowledgeGraph
        
        kg = AdvancedKnowledgeGraph()
        await kg.initialize_models()
        print("✅ Knowledge Graph System initialized")
        
        print("\n🎉 All Advanced ML Components tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Advanced ML Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_project_structure():
    """Ensure proper project structure."""
    
    print("\n📁 Setting up project structure...")
    
    directories = [
        "inference/backend/personalization",
        "models/personalization",
        "data/knowledge_graphs",
        "logs/training",
        "experiments"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("✅ Project structure setup complete!")

def create_test_script():
    """Create a comprehensive test script."""
    
    test_script_content = '''#!/usr/bin/env python3
"""
Comprehensive Test Suite for PeatLearn Advanced ML Features
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "inference" / "backend"))

async def test_complete_pipeline():
    """Test the complete ML pipeline."""
    
    print("🧪 Running Complete Advanced ML Pipeline Test...")
    
    try:
        # Import all components
        from personalization.neural_personalization import (
            AdvancedPersonalizationEngine,
            UserInteraction,
            LearningState
        )
        from personalization.rl_agent import (
            AdaptiveLearningAgent,
            LearningEnvironmentState
        )
        from personalization.knowledge_graph import AdvancedKnowledgeGraph
        
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
        
        print("\\n🎉 Complete ML Pipeline Test Passed!")
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
    '''
    
    with open("test_advanced_ml.py", "w") as f:
        f.write(test_script_content)
    
    print("✅ Created comprehensive test script: test_advanced_ml.py")

def main():
    """Main setup function."""
    
    print("🚀 PeatLearn Advanced ML Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Setup project structure
    setup_project_structure()
    
    # Install requirements
    if not install_requirements():
        print("❌ Installation failed")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("❌ Import tests failed")
        sys.exit(1)
    
    # Test advanced components
    try:
        success = asyncio.run(test_advanced_components())
        if not success:
            print("❌ Advanced component tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Could not run advanced component tests: {e}")
        print("This might be expected if dependencies aren't fully installed yet.")
    
    # Create test script
    create_test_script()
    
    print("\n" + "=" * 50)
    print("🎉 PeatLearn Advanced ML Setup Complete!")
    print("\nNext steps:")
    print("1. Run: python test_advanced_ml.py")
    print("2. Start the advanced server: python inference/backend/advanced_app.py")
    print("3. Test the API endpoints")
    print("\nAdvanced features available:")
    print("- Neural Collaborative Filtering")
    print("- Deep Reinforcement Learning")
    print("- Knowledge Graph Neural Networks")
    print("- Multi-task Learning")
    print("- Fine-tuned Domain Embeddings")

if __name__ == "__main__":
    main()

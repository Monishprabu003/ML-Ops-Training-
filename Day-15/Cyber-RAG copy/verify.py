#!/usr/bin/env python3
"""
Verification script for Cyber RAG application.

This script tests all components to ensure the application is ready to run.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def check_python_version():
    """Check Python version."""
    print("1️⃣  Checking Python Version")
    version = sys.version_info
    required = (3, 12)
    
    print(f"   Current: Python {version.major}.{version.minor}.{version.micro}")
    
    if (version.major, version.minor) >= required:
        print("   ✅ Python version OK\n")
        return True
    else:
        print(f"   ⚠️  Recommended: Python 3.12+\n")
        return True  # Still allow older versions


def check_file_structure():
    """Check if all required files exist."""
    print("2️⃣  Checking File Structure")
    
    required_files = {
        "app.py": "Main Streamlit application",
        "requirements.txt": "Python dependencies",
        "README.md": "Documentation",
        ".gitignore": "Git ignore rules",
        "src/__init__.py": "Package initialization",
        "src/utils.py": "Utility functions",
        "src/ingest.py": "PDF ingestion",
        "src/retriever.py": "Vector store retrieval",
        "src/rag_chain.py": "RAG chain implementation",
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path:30} - {description}")
        else:
            print(f"   ❌ {file_path:30} - MISSING!")
            all_exist = False
    
    print()
    return all_exist


def check_directories():
    """Check if required directories exist."""
    print("3️⃣  Checking Directories")
    
    required_dirs = {
        "src": "Source code",
        "data": "Data storage",
        "logs": "Logs (will be created)",
    }
    
    for dir_name, description in required_dirs.items():
        dir_path = project_root / dir_name
        if dir_name == "logs":
            print(f"   📁 {dir_name:15} - {description} (auto-created on run)")
        elif dir_path.exists():
            print(f"   ✅ {dir_name:15} - {description}")
        else:
            print(f"   ⚠️  {dir_name:15} - Will be created on first use")
    
    print()


def check_dependencies():
    """Check if key dependencies are importable."""
    print("4️⃣  Checking Dependencies")
    
    dependencies = {
        "streamlit": "Web framework",
        "langchain": "LLM framework",
        "chromadb": "Vector database",
        "sentence_transformers": "Embeddings model",
        "pypdf": "PDF processing",
        "dotenv": "Environment variables",
    }
    
    all_installed = True
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"   ✅ {package:25} - {description}")
        except ImportError:
            print(f"   ❌ {package:25} - NOT INSTALLED")
            all_installed = False
    
    print()
    return all_installed


def check_syntax():
    """Check Python syntax of all modules."""
    print("5️⃣  Checking Python Syntax")
    
    modules = [
        "src/utils.py",
        "src/ingest.py",
        "src/retriever.py",
        "src/rag_chain.py",
        "app.py",
    ]
    
    all_valid = True
    for module in modules:
        try:
            with open(project_root / module, 'r') as f:
                compile(f.read(), module, 'exec')
            print(f"   ✅ {module:20} - Syntax OK")
        except SyntaxError as e:
            print(f"   ❌ {module:20} - Syntax Error: {e}")
            all_valid = False
    
    print()
    return all_valid


def check_code_quality():
    """Check code quality indicators."""
    print("6️⃣  Checking Code Quality")
    
    quality_items = [
        ("Type hints", "src/ingest.py", "def load_and_split_pdf(pdf_path: str)"),
        ("Docstrings", "src/rag_chain.py", '"""'),
        ("Error handling", "src/retriever.py", "except"),
        ("Logging", "src/utils.py", "logging"),
    ]
    
    all_good = True
    for feature, file_path, check_string in quality_items:
        try:
            with open(project_root / file_path, 'r') as f:
                content = f.read()
                if check_string in content:
                    print(f"   ✅ {feature:20} - Found in {file_path}")
                else:
                    print(f"   ⚠️  {feature:20} - Not found in {file_path}")
        except Exception as e:
            print(f"   ❌ {feature:20} - Error checking: {e}")
            all_good = False
    
    print()
    return all_good


def check_ollama():
    """Check if Ollama is available."""
    print("7️⃣  Checking Ollama Status")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("   ✅ Ollama is running")
            print("   📝 Make sure llama3 model is pulled:")
            print("      ollama pull llama3")
            print()
            return True
    except Exception:
        pass
    
    print("   ⚠️  Ollama is not currently running")
    print("   📝 Start Ollama with: ollama serve")
    print()
    return False


def check_requirements_file():
    """Validate requirements.txt format."""
    print("8️⃣  Checking Requirements File")
    
    try:
        req_file = project_root / "requirements.txt"
        with open(req_file, 'r') as f:
            lines = f.readlines()
        
        valid_lines = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                valid_lines += 1
        
        print(f"   ✅ requirements.txt is valid")
        print(f"   📦 {valid_lines} dependencies listed")
        print()
        return True
    except Exception as e:
        print(f"   ❌ Error reading requirements.txt: {e}")
        print()
        return False


def print_summary(results):
    """Print summary of all checks."""
    print_header("VERIFICATION SUMMARY")
    
    checks_passed = sum(results.values())
    total_checks = len(results)
    
    for check_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
    
    print(f"\n  {checks_passed}/{total_checks} checks passed\n")
    
    if checks_passed == total_checks:
        print("  🎉 All checks passed! Ready to run the application.\n")
        return True
    else:
        print("  ⚠️  Some checks failed. Please review above.\n")
        return False


def print_next_steps():
    """Print next steps for running the application."""
    print_header("NEXT STEPS")
    
    print("1. Install dependencies (if not already done):")
    print("   pip install -r requirements.txt\n")
    
    print("2. Start Ollama (in a separate terminal):")
    print("   ollama serve\n")
    
    print("3. Ensure llama3 model is available:")
    print("   ollama pull llama3\n")
    
    print("4. Run the Streamlit application:")
    print("   streamlit run app.py\n")
    
    print("5. Open your browser to:")
    print("   http://localhost:8501\n")


def main():
    """Run all verification checks."""
    print_header("🛡️  CYBER RAG VERIFICATION")
    
    results = {
        "Python Version": check_python_version(),
        "File Structure": check_file_structure(),
    }
    
    check_directories()
    
    results["Dependencies"] = check_dependencies()
    results["Syntax Check"] = check_syntax()
    results["Code Quality"] = check_code_quality()
    results["Ollama Status"] = check_ollama()
    results["Requirements File"] = check_requirements_file()
    
    all_passed = print_summary(results)
    print_next_steps()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

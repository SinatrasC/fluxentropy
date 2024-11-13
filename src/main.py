import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import utils
from engine import Engine

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
if device.type == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    seed = 1337
    torch.manual_seed(seed)
    # model_id = 'HuggingFaceTB/SmolLM-360M-Instruct'
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"\nLoading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Model loaded successfully")

    print("\nInitializing Engine...")
    engine = Engine(model, tokenizer, device=device, dtype=torch.bfloat16)
    print("Engine initialized")

    # Example inputs
    input_strings = [
    "The quick brown fox jumps over the lazy dog.",  # Classic pangram
    "In quantum mechanics, particles can exist in multiple states simultaneously.",  # Scientific
    "她站在窗前，望着远方的山峰。",  # Chinese (Looking at distant mountains)
    "To be, or not to be, that is the question.",  # Literary/Shakespeare
    "The cryptocurrency market experienced significant volatility today.",  # Financial news
    "Je pense, donc je suis.",  # French philosophy (Descartes)
    "🌟 Dancing under the moonlight, spirits high and hearts light. 🌙",  # Emojis and poetic
    "SELECT * FROM users WHERE age > 18;",  # SQL code
    "The neural network achieved 98.5% accuracy on the test dataset.",  # AI/ML
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",  # Latin placeholder
    "Breaking: Major breakthrough in fusion energy announced today!",  # News headline
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",  # Python code
    "Step 1: Preheat oven to 350°F. Step 2: Mix ingredients thoroughly.",  # Recipe instructions
    "Once upon a time, in a galaxy far, far away...",  # Story opening
    "Error 404: Page not found. Please check the URL and try again.",  # Technical error
    "Climate change threatens biodiversity in coral reef ecosystems.",  # Environmental
    "おはようございます、今日はいい天気ですね。",  # Japanese (Good morning, nice weather today)
    "1234567890 !@#$%^&*()_+ <>?:\"{}|",  # Numbers and special characters
    "URGENT: Meeting rescheduled to 3PM EST - All hands required",  # Business communication
    "The composition of Bach's fugues demonstrates mathematical precision.",  # Music analysis
    "Das Leben ist wie ein Fahrrad. Man muss sich vorwärts bewegen.",  # German (Einstein quote)
    "for i in range(len(array)): if array[i] > max_val: max_val = array[i]",  # More Python code
    "CREATE TABLE employees (id INT PRIMARY KEY, name VARCHAR(255));",  # SQL DDL
    "La vita è bella quando si vive con passione.",  # Italian (Life is beautiful...)
    "RT @SpaceX: Successful launch of Starship prototype #42! 🚀",  # Social media
    "В тихом омуте черти водятся.",  # Russian proverb
    "async function fetchData() { const response = await fetch(url); }",  # JavaScript async
    "🎮 Level Up! You've earned 1000 XP and unlocked new achievements! 🏆",  # Gaming with emojis
    "<!DOCTYPE html><html><head><title>Hello World</title></head></html>",  # HTML
    "Hola mundo, ¿cómo estás hoy?",  # Spanish greeting
    "import numpy as np; X = np.array([[1, 2], [3, 4]])",  # Scientific Python
    "Breaking News: Artificial Intelligence Achieves New Milestone in Protein Folding",  # Science news
    "public class HelloWorld { public static void main(String[] args) {} }",  # Java
    "The mitochondria is the powerhouse of the cell.",  # Biology
    "git commit -m \"Fix: resolve memory leak in main loop\"",  # Git command
    "अतिथि देवो भव:",  # Sanskrit (Guest is God)
    "try { throw new Error('Test'); } catch (e) { console.log(e); }",  # JavaScript error handling
    "Dans les champs de l'observation, le hasard ne favorise que les esprits préparés.",  # French (Pasteur)
    "docker run -d -p 80:80 nginx:latest",  # Docker command
    "While(true) { System.out.println(\"Hello, World!\"); }",  # Infinite loop
    "kubectl get pods -n kubernetes-dashboard",  # Kubernetes command
    "Χαίρετε! Πώς είστε σήμερα;",  # Greek greeting
    "const handleSubmit = (e) => { e.preventDefault(); setState(newValue); };",  # React code
    "مرحبا بالعالم",  # Arabic (Hello World)
    "SELECT COUNT(*) OVER (PARTITION BY department) FROM employees;",  # Advanced SQL
    "pip install tensorflow==2.8.0 torch==2.0.0 transformers==4.28.0",  # Package installation
    "한글은 세상에서 가장 과학적인 글자입니다.",  # Korean (Hangul is the most scientific writing system)
    "{ \"name\": \"John\", \"age\": 30, \"city\": \"New York\" }",  # JSON data
    "CRITICAL: Memory usage exceeded 90% threshold at 02:45:30 UTC",  # System log
    "@media (max-width: 768px) { .container { flex-direction: column; } }",  # CSS media query
    "Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...",  # Mathematical sequence
    "$ curl -X POST https://api.example.com/v1/data -H \"Content-Type: application/json\"",  # CURL command
    "WARNING: Certificate expires in 7 days. Please renew SSL certificate.",  # Security warning
    "sudo apt-get update && sudo apt-get upgrade -y",  # Linux command
    "print(f\"Current temperature: {temp:.2f}°C at {time:%H:%M:%S}\")",  # Python f-string
    "Революция в квантовых вычислениях: создан 1000-кубитный процессор",  # Russian tech news
    "interface User { id: string; name: string; age: number; }",  # TypeScript interface
    "O Romeo, Romeo! wherefore art thou Romeo?",  # Shakespeare quote
    "Exception in thread \"main\" java.lang.NullPointerException at Main.java:42",  # Java error
    "今日は富士山に登りました。頂上からの景色は素晴らしかったです。"  # Japanese (Climbing Mt. Fuji)
    ]
    # Define configuration for per-token analysis
    config_per_token = {
        "mechanism": "per_token",          # Options: "per_token", "per_string"
        "compute_entropy": True,
        "compute_varentropy": True,
        "output_format": "dict"            # Options: "dict", "tensor", "list"
    }

    # Define configuration for per-string analysis
    config_per_string = {
        "mechanism": "per_string",
        "compute_entropy": True,
        "compute_varentropy": True,
        "output_format": "dict"
    }

    print(f"\nProcessing {len(input_strings)} input strings...")
    print("Sample of first 3 inputs:")
    for i, text in enumerate(input_strings[:3]):
        print(f"{i+1}. {text[:100]}...")

    print("\nComputing per-token characteristics...")
    results_per_token = engine.compute_characteristics(
        input_strings=input_strings,
        config=config_per_token,
        max_length=512
    )
    print("Per-token analysis complete")
    print(f"Results shape: {type(results_per_token)}")
    if isinstance(results_per_token, dict):
        print("Keys:", results_per_token.keys())

    print("\nComputing per-string characteristics...")
    results_full = engine.compute_characteristics(
        input_strings=input_strings,
        config=config_per_string,
        max_length=512
    )
    print("Per-string analysis complete")
    print(f"Results shape: {type(results_full)}")
    if isinstance(results_full, dict):
        print("Keys:", results_full.keys())

    print("\nGenerating visualizations...")
    fig_per_token = utils.visualize_results(
        results=results_per_token,
        config=config_per_token,
        title="Entropy Analysis (Per Token)"
    )
    print("Per-token visualization complete")
    fig_per_token.show()

    fig_full = utils.visualize_results(
        results=results_full,
        config=config_per_string,
        title="Entropy Analysis (Full String)"
    )
    print("Per-string visualization complete")
    fig_full.show()

    print("\nPermuting dataset...")
    permuted_dataset, sorted_characteristics = engine.permute_dataset(
        dataset=input_strings,
        config=config_per_token,
        sort_by="entropy_token_avg",
        descending=True,
        max_length=512
    )
    print("Dataset permutation complete")

    print("\nTop 5 highest entropy strings:")
    for idx, (string, entropy) in enumerate(zip(permuted_dataset[:5], sorted_characteristics[:5])):
        print(f"{idx + 1}: {entropy:.4f} - {string[:100]}...")

    print("\nBottom 5 lowest entropy strings:")
    for idx, (string, entropy) in enumerate(zip(permuted_dataset[-5:], sorted_characteristics[-5:])):
        print(f"{len(permuted_dataset) - 4 + idx}: {entropy:.4f} - {string[:100]}...")

    print("\nGenerating sorted visualization...")
    sorted_results = engine.compute_characteristics(
        input_strings=permuted_dataset,
        config=config_per_token,
        max_length=512
    )
    sorted_fig = utils.visualize_results(
        results=sorted_results,
        config=config_per_token,
        title="Sorted Entropy Analysis (Per Token)"
    )
    print("Sorted visualization complete")
    sorted_fig.show()

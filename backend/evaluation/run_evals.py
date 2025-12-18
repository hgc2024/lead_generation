import os
import time
import sys
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style, init
from langchain_ollama import ChatOllama

# Add parent directory to path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend import rag_service
from backend.evaluation.jury import JURY_PANEL

init(autoreset=True)

def run_evaluation(num_samples=5):
    print(f"{Fore.CYAN}=======================================================")
    print(f"{Fore.CYAN}   Hybrid AI Sales Agent - Corporate Jury Evaluation   ")
    print(f"{Fore.CYAN}======================================================={Style.RESET_ALL}")
    print(f"Running on recognized hardware (Ollama auto-detection).")
    print(f"Jury Panel: {[j.name for j in JURY_PANEL]}")
    print(f"Samples: {num_samples} Random Leads\n")

    # Load Data directly
    try:
        # Resolve absolute path: run_evals.py -> evaluation -> backend -> lead_generation(inner) -> lead_generation(outer) -> data
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/archive/Lead Scoring.csv'))
        df = pd.read_csv(data_path)
        # Filter for high quality leads to make simulation realistic
        high_value_leads = df[df['Converted'] == 1].sample(num_samples).to_dict(orient='records')
    except Exception as e:
        print(f"{Fore.RED}Error loading data: {e}")
        return

    report_lines = []
    report_lines.append(f"# Evaluation Report\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    grand_total_score = 0
    total_latency = 0
    total_evals = 0

    for i, lead in enumerate(high_value_leads):
        print(f"\n{Fore.YELLOW}Processing Lead #{i+1} (Source: {lead.get('Lead Source')})...{Style.RESET_ALL}")
        
        # 1. Measure Latency & Generate
        start_time = time.time()
        
        # Extract profile manually as we are bypassing the API layer
        lead_profile = {
            'Lead Source': lead.get('Lead Source'),
            'Tags': lead.get('Tags'),
            'Total Time Spent on Website': lead.get('Total Time Spent on Website', 0),
            'Specialization': lead.get('Specialization', '')
        }
        
        result = rag_service.run_rag_pipeline(lead_profile)
        
        end_time = time.time()
        latency = end_time - start_time
        total_latency += latency
        
        email_draft = result['email_draft']
        product = result['recommended_product']
        
        print(f"  -> Generated Email in {latency:.2f}s")
        print(f"  -> Product Pitched: {product}")
        
        # 2. Jury Deliberation declaration
        report_lines.append(f"## Lead #{i+1}: {lead.get('Lead Source')}\n")
        report_lines.append(f"**Recommended Product**: {product}\n")
        report_lines.append(f"**Latency**: {latency:.2f}s\n")
        report_lines.append(f"### Email Draft:\n```\n{email_draft}\n```\n")
        report_lines.append(f"### Jury Scores:\n")
        
        print(f"  -> {Fore.MAGENTA}Jury Deliberating...{Style.RESET_ALL}")
        
        lead_scores = []
        
        for jurist in JURY_PANEL:
            eval_result = jurist.evaluate(email_draft, lead_profile, product)
            score = eval_result.get('score', 0)
            reason = eval_result.get('reasoning', 'No reasoning provided')
            
            lead_scores.append(score)
            
            # Console Output
            color = Fore.GREEN if score >= 8 else (Fore.YELLOW if score >= 5 else Fore.RED)
            print(f"    - {jurist.name}: {color}{score}/10{Style.RESET_ALL} | {reason}")
            
            # Report Output
            report_lines.append(f"- **{jurist.name}** ({score}/10): {reason}\n")
            
        avg_score = sum(lead_scores) / len(lead_scores)
        grand_total_score += avg_score
        total_evals += 1
        
        print(f"  -> {Fore.CYAN}Average Score: {avg_score:.1f}/10{Style.RESET_ALL}")
        report_lines.append(f"\n**Average Score**: {avg_score:.1f}/10\n\n---\n")

    # Final Stats
    avg_latency = total_latency / num_samples
    overall_avg_score = grand_total_score / total_evals if total_evals > 0 else 0
    
    print(f"\n{Fore.GREEN}======================================================={Style.RESET_ALL}")
    print(f"EVALUATION COMPLETE")
    print(f"Average Latency: {avg_latency:.2f}s")
    print(f"Overall Quality Score: {overall_avg_score:.2f}/10")
    print(f"Detailed Report saved to: evaluation_report.md")
    
    report_lines.insert(0, f"**Overall Quality Score**: {overall_avg_score:.2f}/10\n**Average Latency**: {avg_latency:.2f}s\n\n")

    with open('evaluation_report.md', 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

if __name__ == "__main__":
    # Check for Ollama
    try:
        print("Checking Ollama connection...")
        # Simple test
        test = ChatOllama(model="mistral").invoke("hello")
        print(f"{Fore.GREEN}Ollama Online.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}CRITICAL: Ollama is not accessible. Please run 'ollama serve' in a separate terminal.{Style.RESET_ALL}") 
        print(str(e))
        sys.exit(1)
        
    run_evaluation()

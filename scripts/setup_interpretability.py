from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def run(*args): subprocess.check_call([sys.executable,'-m','pip','install',*args])
if __name__=='__main__':
    run('-r',str(ROOT/'requirements.txt'))
    run('-r',str(ROOT/'requirements-model.txt'))
    run('-r',str(ROOT/'requirements-interpretability.txt'))
    print('AI Sentinel Lab interpretability environment is ready.')
    print('Next: python -m interpretability.run_research --model gpt2')

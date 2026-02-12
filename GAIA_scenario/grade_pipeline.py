import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from grader import AnswerGrader
from llmforall import get_llm_config


class GradingPipeline:
    """Pipeline to grade all agent answers in a comparison file."""
    
    def __init__(self, grader_model: dict):
        if "host.docker.internal" in grader_model["base_url"]:
            print("Note: Grader is configured to use host.docker.internal for LLM proxy.")
            print("Switching to localhost for compatibility.")
            grader_model["base_url"] = grader_model["base_url"].replace("host.docker.internal", "localhost")
        self.grader = AnswerGrader(model=grader_model)
        self.model_name = grader_model['model']
    
    def _grade_questions(self, questions: list, framework_name: Optional[str] = None) -> tuple[list, int, int]:
        """
        Grade a list of questions and return updated questions with grading results.
        
        Args:
            questions: List of question dictionaries
            framework_name: Optional name for progress display
            
        Returns:
            Tuple of (graded_questions, correct_count, graded_count)
        """
        correct_count = 0
        graded_count = 0
        graded_questions = []
        
        prefix = f"  " if framework_name else ""
        
        for i, question_data in enumerate(questions):
            print(f"{prefix}Question {i+1}/{len(questions)}", end="\r")
            
            grade_result = self.grader.grade_answer(
                agent_answer=question_data["agent_answer"],
                correct_answer=question_data["correct_answer"],
                question=question_data["question"]
            )
            
            graded_question = question_data.copy()
            graded_question["grading"] = grade_result
            graded_questions.append(graded_question)
            
            if question_data["agent_answer"] is not None:
                graded_count += 1
                if grade_result["is_correct"]:
                    correct_count += 1
        
        return graded_questions, correct_count, graded_count
    
    def _create_grading_summary(self, questions: list, correct_count: int, graded_count: int) -> Dict:
        """Create grading summary with metrics and literary details."""
        summary = {
            "total_questions": len(questions),
            "graded_questions": graded_count,
            "correct_answers": correct_count,
            "accuracy": correct_count / graded_count if graded_count > 0 else 0,
            "grader_model": self.model_name,
            "graded_at": datetime.now().isoformat()
        }
        
        print("Calculating literary details...")
        summary["literary_details"] = self.grader.access_preformance(questions, summary)
        
        return summary
    
    def _save_results(self, data: Dict, input_path: str, output_dir: Optional[str], subdir: str = "") -> str:
        """Save graded results to file."""
        if output_dir is None:
            input_path_obj = Path(input_path)
            if subdir:
                output_dir = str(input_path_obj.parent / f"graded/{subdir}{input_path_obj.stem}_graded{input_path_obj.suffix}")
            else:
                output_dir = str(input_path_obj.parent / f"{input_path_obj.stem}_graded{input_path_obj.suffix}")
        
        else:
            output_path_obj = Path(output_dir)
            if output_path_obj.is_dir():
                if subdir:
                    output_dir = str(output_path_obj / f"graded/{subdir}{Path(input_path).stem}_graded{Path(input_path).suffix}")
                else:
                    output_dir = str(output_path_obj / f"{Path(input_path).stem}_graded{Path(input_path).suffix}")
            else:
                output_dir = str(output_path_obj)
        
        with open(output_dir, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_dir
        
    def grade_comparison_file(self, input_path: str, output_path: Optional[str] = None, subdir: str = "") -> Dict:
        """
        Grade all answers in a comparison JSON file.
        
        Args:
            input_path: Path to comparison JSON file
            output_path: Optional path for graded output (defaults to adding _graded suffix)
            subdir: Subdirectory within graded/ folder
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        graded_data = {}
        
        for framework, framework_data in data.items():
            print(f"\nGrading {framework}...")
            
            graded_questions, correct_count, graded_count = self._grade_questions(
                framework_data["questions"], 
                framework
            )
            
            graded_framework = framework_data.copy()
            graded_framework["questions"] = graded_questions
            graded_framework["grading_summary"] = self._create_grading_summary(
                framework_data["questions"],
                correct_count,
                graded_count
            )
            
            graded_data[framework] = graded_framework
            
            if graded_count > 0:
                accuracy = correct_count / graded_count * 100
                print(f"  {framework}: {correct_count}/{graded_count} correct ({accuracy:.1f}%)")
            else:
                print(f"  {framework}: No answers to grade.")
        
        output_file = self._save_results(graded_data, input_path, output_path, subdir)
        print(f"\nGraded results saved to: {output_file}")
        return graded_data

    def grade_single_agent(self, input_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Grade all answers in a single-agent JSON file.
        
        Args:
            input_path: Path to single-agent JSON file
            output_path: Optional path for graded output (defaults to adding _graded suffix)
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        print("Grading single agent...")
        
        graded_questions, correct_count, graded_count = self._grade_questions(data["questions"])
        
        data["questions"] = graded_questions
        data["grading_summary"] = self._create_grading_summary(
            data["questions"],
            correct_count,
            graded_count
        )
        
        if graded_count > 0:
            accuracy = correct_count / graded_count * 100
            print(f"  Correct Answers: {correct_count}/{graded_count} ({accuracy:.1f}%)")
        else:
            print("  No answers to grade.")
        
        output_file = self._save_results(data, input_path, output_path)
        print(f"\nGraded results saved to: {output_file}")
        return data

def fix_assessment(input_path: str, model_idx: int = 10):
    """Fix literary details assessment in already graded files."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    pipeline = GradingPipeline(grader_model=get_llm_config(model_idx))
    
    for _, framework_data in data.items():
        framework_data["grading_summary"]["literary_details"] = pipeline.grader.access_preformance(
            framework_data["questions"],
            framework_data["grading_summary"]
        )
    
    with open(input_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed assessment in: {input_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python grade_pipeline.py <input_path> [output_path] [mode]")
        print("  mode: 'single' for single agent, 'fix' for fixing assessment, or omit for comparison")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    mode = sys.argv[3] if len(sys.argv) > 3 else "comparison"
    
    # Determine mode
    if input_path.endswith("_graded.json") or mode == "fix":
        fix_assessment(input_path)
    elif mode == "single":
        pipeline = GradingPipeline(grader_model=get_llm_config(10))
        pipeline.grade_single_agent(input_path, output_path)
    else:
        # Default to comparison mode
        pipeline = GradingPipeline(grader_model=get_llm_config(10))
        if input_path.endswith(".json"):
            pipeline.grade_comparison_file(input_path, output_path)
        else:
            # Input is a directory - process all JSON files
            input_dir = Path(input_path)
            if not input_dir.is_dir():
                print(f"Error: {input_path} is not a valid file or directory")
                sys.exit(1)

            json_files = list(input_dir.glob("*.json"))
            if not json_files:
                print(f"No JSON files found in {input_path}")
                sys.exit(1)

            print(f"Found {len(json_files)} JSON files to process")
            for json_file in json_files:
                print(f"\n{'='*60}")
                print(f"Processing: {json_file.name}")
                print('='*60)
                if output_path:
                    output_dir = Path(output_path)
                    if not Path(output_dir).exists():
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                try:
                    pipeline.grade_comparison_file(str(json_file), output_path)
                except Exception as e:
                    print(f"Error processing {json_file.name}: {str(e)}")

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
        
    def grade_comparison_file(self, input_path: str, output_path: Optional[str] = None, subdir: str = "") -> Dict:
        """
        Grade all answers in a comparison JSON file.
        
        Args:
            input_path: Path to comparison JSON file
            output_path: Optional path for graded output (defaults to adding _graded suffix)
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        graded_data = {}
        
        for framework, framework_data in data.items():
            print(f"\nGrading {framework}...")
            graded_framework = framework_data.copy()
            
            correct_count = 0
            graded_count = 0
            
            for i, question_data in enumerate(framework_data["questions"]):
                print(f"  Question {i+1}/{len(framework_data['questions'])}", end="\r")
                
                grade_result = self.grader.grade_answer(
                    agent_answer=question_data["agent_answer"],
                    correct_answer=question_data["correct_answer"],
                    question=question_data["question"]
                )
                
                graded_framework["questions"][i]["grading"] = grade_result
                
                if question_data["agent_answer"] is not None:
                    graded_count += 1
                    if grade_result["is_correct"]:
                        correct_count += 1
            
            # Add grading summary
            graded_framework["grading_summary"] = {
                "total_questions": len(framework_data["questions"]),
                "graded_questions": graded_count,
                "correct_answers": correct_count,
                "accuracy": correct_count / graded_count if graded_count > 0 else 0,
                "grader_model": self.model_name,
                "graded_at": datetime.now().isoformat()
            }

            print("Calculating literary details...")
            graded_framework["grading_summary"]["literary_details"] = self.grader.access_preformance(
                framework_data["questions"],
                graded_framework["grading_summary"]
            )
            
            graded_data[framework] = graded_framework
            if graded_count > 0:
                print(f"  {framework}: {correct_count}/{graded_count} correct ({correct_count/graded_count*100:.1f}%)")
            else:
                print(f"  {framework}: No answers to grade.")
        
        # Save graded results
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"graded/{subdir}{input_path_obj.stem}_graded{input_path_obj.suffix}")
        
        with open(output_path, 'w') as f:
            json.dump(graded_data, f, indent=2)
        
        print(f"\nGraded results saved to: {output_path}")
        return graded_data

def fix_accessment(input_path: str):
    with open(input_path, 'r') as f:
        data = json.load(f)
    grader = GradingPipeline(grader_model=get_llm_config())
    for _, framework_data in data.items():
        framework_data["grading_summary"]["literary_details"] = grader.grader.access_preformance(
            framework_data["questions"],
            framework_data["grading_summary"]
        )
    with open(input_path, 'w') as f:
        json.dump(data, f, indent=2)
    
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python grade_pipeline.py <comparison_json_path> [output_path]")
        sys.exit(1)
    if sys.argv[1].endswith("_graded.json"):
        fix_accessment(sys.argv[1])
        sys.exit(0)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    pipeline = GradingPipeline(grader_model=get_llm_config(10))
    pipeline.grade_comparison_file(input_path, output_path, subdir="lvl3/")

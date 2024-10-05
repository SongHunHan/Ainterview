import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from schemas.job_post import JobPost, InterviewResponse

class InterviewService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_paths = self.get_model_paths()

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
    def get_model_paths(self):
        base_path = os.path.abspath(os.path.dirname(__file__))
        return {
            'question': [
                os.path.join(base_path, 'model_path', 'question', 'make_all_question'),
                os.path.join(base_path, 'model_path', 'question', 'make_develope_question'),
                os.path.join(base_path, 'model_path', 'question', 'make_normal_question'),
            ],
            'answer': [
                os.path.join(base_path, 'model_path', 'answer', 'make_all_answer'),
                os.path.join(base_path, 'model_path', 'answer', 'make_develope_answer'),
                os.path.join(base_path, 'model_path', 'answer', 'make_normal_answer'),
            ]
        }
    
    def load_model(self, path):
        config = PeftConfig.from_pretrained(path)

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=self.bnb_config,
            attn_implementation="eager",
            device_map='cuda',
            torch_dtype='float16',
        )

        model = PeftModel.from_pretrained(model, path)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    
    def generate_text(self, input_text, model, tokenizer):
        model.to(self.device)
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(self.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=1024,
                num_return_sequences=1,
                do_sample=True,
                num_beams=2,
                top_p=0.95,
                repetition_penalty=1.2,
                temperature=0.5,
                early_stopping=True,
            )
        return_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if "model##" in return_text:
            return return_text.split("model##", 1)[1].strip()
        else:
            return return_text.strip()
    
    def generate_interview_qa(self, job_post: JobPost, model_index: int) -> InterviewResponse:
        question_prompt = """<bos><start_of_turn>user
        당신은 다양한 IT 분야의 전문성을 갖춘 채용 담당자입니다.
        주어진 채용 공고를 바탕으로 가장 핵심적이고 적절한 면접 질문 하나를 생성해야 합니다.
        지원자의 문제 해결 능력과 창의성을 평가할 수 있는 질문을 만들어주세요.

        - 복합적인 질문이 아닌, 주제를 관통하는 하나의 질문을 생성해주세요.
        채용공고 내용: 
        {input_text}<end_of_turn>
        <start_of_turn>model##
        """

        answer_prompt = """<bos><start_of_turn>user
        당신은 대규모 IT 프로젝트를 이끄는 시니어 개발 책임자입니다.
        주어진 채용 공고를 바탕으로 가장 핵심적이고 적절한 면접 답변 하나를 생성해야 합니다.
        지원자의 문제 해결 능력과 창의성을 평가할 수 있는 답변을 만들어주세요.
        채용공고 내용: 
        {input_text}
        
        질문: {question_text}<end_of_turn>
        <start_of_turn>model##
        """

        try: 
            question_model, question_tokenizer = self.load_model(self.model_paths['question'][model_index])

            question = self.generate_text(
                question_prompt.format(input_text=job_post.content),
                question_model,
                question_tokenizer
            )

            answer_model, answer_tokenizer = self.load_model(self.model_paths['answer'][model_index])
        
            answer = self.generate_text(
                answer_prompt.format(input_text=job_post.content, question_text=question),
                answer_model,
                answer_tokenizer
            )

            return InterviewResponse(question=question, answer=answer)
        finally:
            torch.cuda.empty_cache() # CUDA 캐시 정리
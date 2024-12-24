import argparse
import os
import warnings
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from halo import Halo
from colorama import init, Fore
from deepmultilingualpunctuation import PunctuationModel
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_openai import AzureChatOpenAI


warnings.filterwarnings("ignore", category=UserWarning, message="`grouped_entities` is deprecated.")
init(autoreset=True)
load_dotenv()


class Config:
    OPENAI_API_KEY = os.environ['AZURE_OPENAI_API_KEY']
    ENDPOINT = os.environ['AZURE_OPENAI_ENDPOINT']
    DEPLOYMENT_NAME = "gpt-4o-mini"
    API_VERSION = "2024-08-01-preview"

class YouTubeTranscriptHandler:
    punctuation_model = PunctuationModel()

    @staticmethod
    def extract_video_id(url: str) -> str:
        parsed_url = urlparse(url)
        if parsed_url.hostname == 'www.youtube.com':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        return None

    def get_transcript(self, video_id: str) -> str:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([entry['text'] for entry in transcript])

    def add_punctuation(self, text: str) -> str:
        punctuated_text = self.punctuation_model.restore_punctuation(text)
        return punctuated_text.replace('. ', '.\n').replace(', ', ',\n')


class LLMHandler:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            deployment_name=Config.DEPLOYMENT_NAME,
            openai_api_version=Config.API_VERSION,
            openai_api_key=Config.OPENAI_API_KEY,
            azure_endpoint=Config.ENDPOINT,
            openai_api_type="azure"
        )

    def prepare_prompt(self, text: str, question: str, chat_history: list) -> str:
        history = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
        return f"""
        You are an expert answering based only on the provided video transcript.
        Transcript: 

        {text}

        Chat history:
        {history}

        Question: {question}

        Answer in detail using only the transcript and chat history.
        If a question is irrelevant to the video, answer with "I'm sorry, I can't answer that question as it is not relevant to the video content."
        If the answer is absent in the video, respond with "I'm sorry, I can't answer that question as the answer is not present in the video content."
        """

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content.strip()
    
class VideoCLIApp:
    def __init__(self):
        self._transcript_handler = YouTubeTranscriptHandler()
        self._llm_handler = LLMHandler()
        self._chat_history = []
        self._transcript = ""
        self._menu_options = {
            "1": ("Print Punctuated Transcript", self._print_transcript),
            "2": ("Summarize Transcript", self._ask_ai_single_question, "provide a summary of the video content"),
            "3": ("Chat with AI Assistant", self._chat_with_ai_loop),
            "4": ("FAQ", self._ask_ai_single_question, "provide a 5 question FAQ based on the video content"),
            "5": ("Table of Contents", self._ask_ai_single_question, "provide a table of contents based on the video content"),
            "6": ("Change Video", self._change_video),
            "0": ("Exit", self._exit_app)
        }

    def _fetch_video_transcript(self, video_url: str) -> str:
        spinner = Halo(text='Fetching transcript...', spinner='dots')
        spinner.start()
        video_id = self._transcript_handler.extract_video_id(video_url)
        transcript = self._transcript_handler.get_transcript(video_id)
        spinner.succeed(Fore.GREEN + 'Transcript fetched.')

        spinner = Halo(text='Adding punctuation...', spinner='dots')
        spinner.start()
        punctuated_transcript = self._transcript_handler.add_punctuation(transcript)
        spinner.succeed(Fore.GREEN + 'Punctuation added.')
        return punctuated_transcript

    def _menu_loop(self):
        while True:
            print(Fore.CYAN + "\nMenu:")
            for key, (description, *_) in self._menu_options.items():
                print(Fore.CYAN + f"{key}. {description}")
                
            choice = input(Fore.YELLOW + "Choose an option: ")

            if choice in self._menu_options:
                _, action, *question = self._menu_options[choice]
                action(question) if question else action()
            else:
                print(Fore.RED + "Invalid choice. Please try again.")

    def _chat_with_ai_loop(self):
        try:
            while True:
                question = input(Fore.YELLOW + "Enter your question (or type 'exit' or CTRL+C to go back): ")
                if question.lower() == 'exit':
                    break
                answer = self._chat_with_ai(self._transcript, question)
                print(Fore.GREEN + "\nAnswer:\n", answer)
        except KeyboardInterrupt:
            print(Fore.RED + "\nChat interrupted. Returning to menu...")

    def _print_transcript(self):
        print(Fore.GREEN + "\nPunctuated Transcript:\n", self._transcript)

    def _change_video(self):
        url = input(Fore.YELLOW + "Please enter the YouTube video URL: ")
        self._transcript = self._fetch_video_transcript(url)

    def _exit_app(self):
        print(Fore.RED + "Exiting...")
        exit()

    def _ask_ai_single_question(self, question: str):
        answer = self._chat_with_ai(self._transcript, question)
        print(answer)

    def _chat_with_ai(self, transcript: str, question: str) -> str:
        spinner = Halo(text='Asking AI...', spinner='dots')
        spinner.start()
        prompt = self._llm_handler.prepare_prompt(transcript, question, self._chat_history)
        answer = self._llm_handler.invoke(prompt)
        spinner.succeed(Fore.GREEN + 'AI Answer:')
        self._chat_history.append((question, answer))
        return answer

    def run(self, video_url):
        try:
            self._transcript = self._fetch_video_transcript(video_url)
            self._menu_loop()
        except KeyboardInterrupt:
            print(Fore.RED + "\nExiting...")
        except Exception as e:
            print(Fore.RED + f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Video Q&A CLI Tool")
    parser.add_argument("--url", nargs='?', help="YouTube video URL")
    args = parser.parse_args()
    video_url = args.url
    if not video_url:
        video_url = input(Fore.YELLOW + "Please enter the YouTube video URL: ")
    
    VideoCLIApp().run(video_url)

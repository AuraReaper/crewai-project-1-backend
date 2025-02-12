from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel

from youtube_gen.tools.SearchYoutubeTool import YoutubeVideoSearchAndDetailsTool

class ResearchItem(BaseModel):
	title : str
	url : str
	view_count : int

class VideoIdea(BaseModel):
	score : int
	video_title : str
	description : str
	video_id : str
	comment_id : str
	research : list[ResearchItem]

class VideoIdeasList(BaseModel):
	video_ideas : list[VideoIdea]

@CrewBase
class YoutubeGen():
	"""YoutubeGen crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def comment_filter_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['comment_filter_agent'],
			verbose=True
		)

	@agent
	def video_idea_generator_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['video_idea_generator_agent'],
			verbose=True
		)
	
	@agent
	def research_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['research_agent'],
			verbose=True
		)
	
	@agent
	def scoring_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['scoring_agent'],
			verbose=True
		)

	@task
	def filter_comments_task(self) -> Task:
		return Task(
			config=self.tasks_config['filter_comments_task'],
		)

	@task
	def generate_video_ideas_task(self) -> Task:
		return Task(
			config=self.tasks_config['generate_video_ideas_task'],
		)
	
	@task
	def research_video_ideas_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_video_ideas_task'],
			tools=[YoutubeVideoSearchAndDetailsTool()]
		)
	
	@task
	def score_video_ideas_task(self) -> Task:
		return Task(
			config=self.tasks_config['score_video_ideas_task'],
			output_pydantic=VideoIdeasList,
		)
	
	
	@crew
	def crew(self) -> Crew:
		"""Creates the YoutubeGen crew"""


		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)

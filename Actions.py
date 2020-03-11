import os

from alsaaudio import Mixer
from pykeyboard import PyKeyboard


class Actions():
	def __init__(self):
		self.amplifier = 1

		self.keyboard = PyKeyboard()

		# step for increasing and decreasing volume
		self.step = 6

	def run(self, action: str) -> None:
		"""run the predicted action
		
		Arguments:
			action {string} -- one of the following : action1, action2, action3, action4, action5, action6, actionAmplifier1, actionAmplifier2, actionAmplifier3, actionAmplifier4
		
		Returns:
			None -- returns what the action method returns
		"""		
		self.action = action

		# if the action is an amplifier
		if 'actionAmplifier' in self.action:
			return self.actionAmplifier(self.action.split('actionAmplifier')[1])

		return getattr(self, self.action)()


	def action1(self):
		self.keyboard.tap_key(' ')
		os.system('notify-send "action1: Key space is tapped"')


	def action2(self):
		sound = Mixer()
		vol = sound.getvolume()[0] # we can take either 0 or 1 it does not matter
		vol += self.step * self.amplifier
		self.amplifier = 1 # reset the amplifier
		# sound.setvolume(vol)
		os.system('notify-send "Volume up to %s"' % vol)


	def action3(self):
		sound = Mixer()
		vol = sound.getvolume()[0] # we can take either 0 or 1 it does not matter
		vol -= self.step * self.amplifier
		self.amplifier = 1 # reset the amplifier
		# sound.setvolume(vol)
		os.system('notify-send "Volume down to %s"' % vol)


	def action4(self):
		os.system('notify-send "salut toi action 4"')


	def action5(self):
		os.system('notify-send "salut toi action 5"')


	def action6(self):
		os.system('notify-send "salut toi action 6"')


	def actionAmplifier(self, times: int) -> None:
		"""Amplifies the action, should be call before the action
		
		Arguments:
			times {int} -- A positive integer that will amplify the action based on the step
				
		Returns:
			None
		"""		
		os.system('notify-send "amplifier %s set waiting for action"' % times)
		self.amplifier = int(times) + 1 # + 1 because the amplifier actions are set from 0

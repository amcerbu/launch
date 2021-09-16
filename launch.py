import mido
import time
import numpy as np
import harmony
from random import random
import pickle

size = 8
n = 12


def save(obj):
	return (obj.__class__, obj.__dict__)

def load(cls, attributes):
    obj = cls.__new__(cls)
    obj.__dict__.update(attributes)
    return obj



# methods for interacting with control surface
class Launchpad:
	static = True

	def globals(self):
		static = False

		keys = np.zeros((size, size), dtype = 'byte')
		for j in range(size):
			keys[:,j] = range(10 * j + 11, 10 * j + size + 11)

		keys_lookup = {keys[x,y] : (x,y) for x in range(size) for y in range(size)}
		
		top = np.zeros(size + 2, dtype = 'byte')
		top[:] = range(90, 100)
		left = np.zeros(size, dtype = 'byte')
		left[:] = range(10, 81, 10)
		right = np.zeros(size, dtype = 'byte')
		right[:] = range(19, 90, 10)
		bottom = np.zeros((2, size), dtype = 'byte')
		bottom[0] = range(1, 9)
		bottom[1] = range(101, 109)

		control = list(bottom[0]) + list(bottom[1]) + list(top) + list(left) + list(right)
		
		names = ['arm', 'mute', 'solo', 'volume', 'pan', 'sends', 'device', 'stop_clip', \
				 'track_1', 'track_2', 'track_3', 'track_4', 'track_5', 'track_6', 'track_7', 'track_8', \
				 'shift', 'left', 'right', 'session', 'note', 'chord', 'custom', 'sequencer', 'projects', 'logo', \
				 'record', 'play', 'fixed_length', 'quantize', 'duplicate', 'clear', 'down', 'up', \
				 'print_to_clip', 'micro_step', 'mutation', 'probability', 'velocity', 'pattern_settings', 'steps', 'patterns']
		
		control_lookup = {control[i] : names[i] for i in range(len(control))}
		control = {v : k for k, v in control_lookup.items()}

		Launchpad.keys = keys
		Launchpad.keys_lookup = keys_lookup
		Launchpad.control = control
		Launchpad.control_lookup = control_lookup

	# the ioport should be a mido port for reading messages and
	# sending messages (e.g. for note illumination)
	# the output should be a mido port for sending note messages (e.g. a synth)
	def __init__(self, ioport, output):
		if self.static: self.globals()

		self.ioport = ioport
		self.output = output
		self.reset()

	def reset(self):
		for note in range(2 ** 7):
			self.ioport.send(mido.Message('note_off', note = note, velocity = 0))
			self.output.send(mido.Message('note_off', note = note, velocity = 0))

	def read(self, message):
		if message.is_cc() and message.control in self.control_lookup: # control messages
			button = self.control_lookup[message.control]
			on = message.value > 0
			return (button, on)

		elif message.type in ['note_on', 'note_off'] and message.note in self.keys_lookup: # note messages
			note, velocity = message.note, message.velocity
			on = bool((message.type == 'note_on') and velocity)
			pad = self.keys_lookup[note]
			return (pad, on, velocity)

	def write(self, pad, color):
		self.ioport.send(mido.Message('note_on', note = self.keys[pad], velocity = color))

	def button(self, control, color):
		self.ioport.send(mido.Message('note_on', note = self.control[control], velocity = color))

	def play(self, pitch, velocity):
		self.output.send(mido.Message('note_on', note = pitch, velocity = velocity))


class Instrument:
	def __init__(self, ioport, output):

		self.L = Launchpad(ioport, output)
		self.running = True

		self.mode = {'keyboard' : Keyboard(self), 'chord' : Chord(self), 'velocity' : Velocity(self)}
		self.application = self.mode['keyboard']

		self.play = lambda pitch, velocity : self.L.play(pitch, self.mode['velocity'].curve[velocity])
		self.button = lambda control, color : self.L.button(control, color)
		self.reset = lambda : self.L.reset()

		self.switch('keyboard')

	def quit(self):
		self.running = False
		self.reset()

	def process(self, message):
		command = self.L.read(message)
		if command is None: return

		if len(command) == 2:
			button, on = command

			if on:
				if button == 'shift': self.quit()
				[self.switch(mode) for mode in self.mode if button == self.mode[mode].button]

		elif len(command) == 3:
			pad, on, velocity = command

		self.application.process(command)

	def switch(self, appname):
		self.application.suspend()
		self.application = self.mode[appname]
		self.application.startup()

	def display(self):
		self.application.print()
		for x in range(size):
			for y in range(size):
				self.L.write((x,y), self.application.display[x,y])


class Application:
	def __init__(self, owner):
		self.owner = owner

		self.display = np.zeros((size, size), dtype = 'byte')
		self.button = None
		self.color = None

		self.piano = np.array([-1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0])
		self.keycolor = np.array([[119, 0, 96], [40, 41, 60], [88, 101, 54]], dtype = 'byte')

		self.bins = {name : i for i, name in enumerate( \
			['track_1', 'track_2', 'track_3', 'track_4', 'track_5', 'track_6', 'track_7', 'track_8', \
			 'arm', 'mute', 'solo', 'volume', 'pan', 'sends', 'device', 'stop_clip'])}

		self.neutral = 0
		self.pressed = 1
		self.triggered = 2

		self.range = range(21, 109)

	def process(self, command):
		pass

	def suspend(self):
		self.owner.button(self.button, 0)

	def startup(self):
		self.owner.button(self.button, self.color)
		self.print()

	def print(self):
		pass


class Keyboard(Application):
	def __init__(self, owner):
		super().__init__(owner)

		self.button = 'note'
		self.color = 40
		self.handedness = 'right' # ambidextrous

		self.tune(36, 5) # C1, fourths tuning
		self.sustains = {(x,y) : set() for x in range(size) for y in range(size)}
		self.sustained = {note : set() for note in self.range}

		self.harmonizes = {(x,y) : set() for x in range(size) for y in range(size)}
		self.harmonized = {note : set() for note in self.range}

		self.history = []
		self.historical = True
		self.window = 5
		self.voicings = harmony.voicings

		self.harmonizer = None
		self.harmony = None
		
		self.rootless = False
		self.walking = False # chord buttons rather than pads trigger harmony (raised an octave if true)
		self.chorale = True # chord qualities are determined procedurally unless otherwise structure is imposed

	def suspend(self):
		super().suspend()

		for pad in (coordinate for coordinate in self.sustains if self.sustains[coordinate]):
			self.release(pad)
			self.unchord(pad, displacement = n if self.walking else 0)

	def tune(self, origin, tuning):
		self.origin = origin
		self.tuning = tuning
		self.breadth = (size - 1) * tuning + size - 1
		self.remap()

	def remap(self):
		self.map = np.array([[self.origin + x + self.tuning * y for y in range(size)] for x in range(size)])

		if self.handedness == 'ambi':
			self.map[:,:size // 2] = self.map[:,:size // 2][::-1]

		elif self.handedness == 'left':
			self.map = self.map[::-1]

	def transpose(self, step):
		if self.origin + step in self.range and self.origin + step + self.breadth in self.range:
			self.origin += step

		self.remap()

	def print(self):
		self.keystate = np.zeros((size, size), dtype = 'byte')
		inscope = [bool(self.sustained[i]) for i in range(self.origin, self.origin + self.breadth + 1)]
		inharm = [bool(self.harmonized[i]) for i in range(self.origin, self.origin + self.breadth + 1)]

		for y in range(size):
			self.keystate[:,y][inscope[self.tuning * y : self.tuning * y + size]] = self.pressed
			self.keystate[:,y][inharm[self.tuning * y : self.tuning * y + size]] = self.triggered
			
			if self.handedness == 'left' or (self.handedness == 'ambi' and y < size // 2):
				self.keystate[:,y] = self.keystate[:,y][::-1]

		self.display = self.keycolor[self.keystate, self.piano[self.map % n]]

	def process(self, command):
		if len(command) == 2:
			button, on = command

			if on and button == 'up': self.transpose(self.tuning)
			elif on and button == 'down': self.transpose(-self.tuning)
			elif on and button == 'left': self.transpose(-1)
			elif on and button == 'right': self.transpose(1)

			elif on and button in self.bins: # a chord button is pressed
				track = self.bins[button]
				self.owner.button(button, 5)

				self.harmonizer = track
				self.harmony = self.owner.saves[track]

				if self.harmony and self.walking:
					for pad in (coordinate for coordinate in self.sustains if self.sustains[coordinate]):
						self.voicelead(pad)
						self.playchord(pad, 32, displacement = n)

			elif not on and button in self.bins: # a chord button is released
				track = self.bins[button]
				self.owner.button(button, 7 if self.owner.saves[track] else 0)

				if self.walking:
					for pad in (coordinate for coordinate in self.sustains if self.sustains[coordinate]):
						self.unchord(pad, displacement = n)
				
				if self.harmonizer == track: # no more chord buttons held down
					self.harmonizer = None
					self.harmony = None

		elif len(command) == 3:
			pad, on, velocity = command

			if on:
				self.sustains[pad].add(self.map[pad]) # record that pad is responsible for note
				self.sustained[self.map[pad]].add(pad) # record that note is held by pad
				
				if not (self.harmony and self.rootless):
					self.owner.play(self.map[pad], 0)
					self.owner.play(self.map[pad], velocity)

				if self.harmony and not self.walking:
					self.voicelead(pad)
					self.playchord(pad, velocity = 32)

			elif self.sustains[pad]:
				self.release(pad)
				self.unchord(pad, displacement = n if self.walking else 0)

	def voicelead(self, pad):
		roots = self.owner.roots[self.harmonizer]
		roots = roots if roots else self.harmony
		condensed = {note % 12 for chord in self.history for note in chord}

		structures = []
		for root in roots:
			structure = [(note - root) % n for note in self.harmony if note != root]
			structures.append(structure)

		options = []
		for structure in structures:
			for voicing in self.voicings[len(structure)]:
				option = harmony.voice(structure, voicing)
				option = [self.map[pad] + note for note in option]
				option = [note for note in option if note in self.range]
				options.append(option)

		last = None
		if self.history:
			last = self.history[-1]
			best = min(options, key = lambda x : self.distance(last, x) + self.historical * len(condensed ^ set(p % 12 for p in x)) ** 2)
		else:
			best = options[int(random() * len(options))]

		self.history.append(best)
		if len(self.history) > self.window:
			self.history.pop(0)

	def playchord(self, pad, velocity, displacement = 0):
		chord = set(pitch + displacement for pitch in self.history[-1])
		chord = set(pitch for pitch in chord if pitch in self.range)

		self.harmonizes[pad].update(chord)
		for pitch in chord:
			self.harmonized[pitch].add(pad)

		for pitch in chord:
			self.owner.play(pitch, velocity)

	def release(self, pad):
		for note in self.sustains[pad]:
			self.sustained[note] -= {pad}
			if not len(self.sustained[note] | self.harmonized[note]):
				self.owner.play(note, 0)

		self.sustains[pad] = set()

	def unchord(self, pad, displacement = 0):
		for note in self.harmonizes[pad]:
			self.harmonized[note] -= {pad}
			if not len(self.sustained[note] | self.harmonized[note]):
				self.owner.play(note, 0)

		self.harmonizes[pad] = set()

	def distance(self, class1, class2):
		if len(class1) < len(class2):
			return min(self.distance(class1[:i+1] + class1[i:], class2) for i in range(len(class1)))

		elif len(class1) > len(class2):
			return self.distance(class2, class1)

		distance = 0
		for i in range(len(class1)):
			distance += self.penalty(abs(class1[i] - class2[i]))
		
		return distance

	def penalty(self, gap):
		return gap ** 2


class Chord(Application):
	def __init__(self, owner):
		super().__init__(owner)

		self.button = 'chord'
		self.color = 88

		self.clock = [(6,4), (5,5), (4,6), (3,6), (2,5), (1,4), (1,3), (2,2), (3,1), (4,1), (5,2), (6,3)]
		self.keycolor[0,1] = 1

		self.registers = [[i for i in self.range if i % n == k] for k in range(n)]
		for i in [-3, -2, -1, 0]: self.registers[i].pop(0)
		
		self.owner.saves = [set() for i in range(2 * size)]
		self.owner.roots = [set() for i in range(2 * size)]
		self.recording = None

		self.rootmode = False
		self.corners = [(0,0), (7,0), (0,7), (7,7)]

		self.sustains = [self.neutral for i in range(len(self.clock))]

	def suspend(self):
		super().suspend()

		self.recording = None
		self.sustains = [self.neutral for i in range(len(self.clock))]
		self.unsound(range(n))

	def print(self):
		for i, pos in enumerate(self.clock):
			self.display[pos] = self.keycolor[self.sustains[i], self.piano[i]]

		if self.recording is not None:
			color = self.pressed if self.rootmode else self.triggered
			for corner in self.corners:
				self.display[corner] = self.keycolor[color, 1]

		else:
			for pad in self.corners: # + self.borders:
				self.display[pad] = 0

	def paint(self, track):
		for pitch in self.owner.saves[track]:
			self.sustains[pitch] = self.triggered

		for pitch in self.owner.roots[track]:
			self.sustains[pitch] = self.pressed

	def unpaint(self, pitches):
		for pitch in pitches:
			self.sustains[pitch] = self.neutral

	def sound(self, track, velocity = 64):
		for pitch in self.owner.saves[track]:
			self.shepard(pitch, 64)

	def unsound(self, pitches):
		for pitch in pitches:
			self.shepard(pitch, 0)

	def process(self, command):
		if len(command) == 2:
			button, on = command

			if on and button in self.bins: # a chord button is pressed
				track = self.bins[button]
				self.owner.button(button, 5)

				if self.recording is not None:
					pitches = self.owner.saves[self.recording]
					self.unsound(pitches)
					self.unpaint(pitches)

				self.sound(track)
				self.paint(track)
				self.recording = track

			elif not on and button in self.bins: # a chord button is released
				track = self.bins[button]
				self.owner.button(button, 7 if self.owner.saves[track] else 0)
				
				if self.recording == track: # no more chord buttons held down
					pitches = range(n)
					self.unsound(pitches)
					self.unpaint(pitches)
					self.recording = None
					self.rootmode = False

			elif on and self.recording is not None and button in ['left', 'right']: # a chord button is held and arrow key pressed
				displacement = 1 if button == 'left' else - 1
				newsaves = set((a + displacement) % n for a in self.owner.saves[self.recording])
				newroots = set((a + displacement) % n for a in self.owner.roots[self.recording])

				togo = self.owner.saves[self.recording] - newsaves
				self.unsound(togo)
				self.unpaint(togo)

				self.owner.saves[self.recording] = newsaves
				self.owner.roots[self.recording] = newroots

				self.sound(self.recording)
				self.paint(self.recording)

		elif len(command) == 3:
			pad, on, velocity = command

			if pad in self.clock:
				note = self.clock.index(pad)

				if self.recording is None: # no chord buttons pressed
					if on:
						self.sustains[note] = self.triggered
						self.shepard(note, velocity)
					else:
						self.sustains[note] = self.neutral
						self.shepard(note, 0)

				else: # at least one chord button pressed
					if self.rootmode: # if in root-editing mode
						if on and note in self.owner.saves[self.recording]:
							self.owner.roots[self.recording] ^= {note}
							if self.sustains[note] == self.triggered:
								self.shepard(note, 0)
								self.shepard(note, velocity)
							else:
								self.sustains[note] = self.triggered
							self.paint(self.recording)

					elif on: # not in root-editing mode; some pad is pressed
						self.owner.saves[self.recording] ^= {note}
						self.owner.roots[self.recording] &= self.owner.saves[self.recording]

						if note not in self.owner.saves[self.recording]:
							self.sustains[note] = self.neutral
							self.shepard(note, 0)

						self.sound(self.recording)
						self.paint(self.recording)

			elif on and self.recording is not None:
				self.rootmode = not self.rootmode

	def shepard(self, note, velocity):
		k = len(self.registers[note])
		for i, pitch in enumerate(self.registers[note]):
			self.owner.play(pitch, int(velocity * (i / k) * (1 - i / k)))


class Velocity(Application):
	def __init__(self, owner):
		super().__init__(owner)

		self.button = 'custom'
		self.color = 48
		self.colors = [48, 49, 50, 51]

		self.tester = None
		self.testcolors = [8, 9, 10, 11]

		self.curve = None
		self.bars = None
		self.reset()

	def startup(self):
		super().startup()
		self.owner.button('clear', self.color)

	def suspend(self):
		super().suspend()
		self.owner.button('clear', 0)

	def reset(self):
		self.curve = np.array([i for i in range(2 ** 7)], dtype = 'byte')
		self.bars = np.array([i for i in range(size)], dtype = 'float')

	def fit(self):
		bars = np.zeros(len(self.bars) + 1, dtype = 'float')
		bars[:-1] = self.bars
		bars[-1] = 2 * bars[-2] - bars[-3]
		skip = 2 ** 7 // size
		for x in range(size):
			for t in range(skip):
				self.curve[x * skip + t] = min(2 ** 7 - 1, max(0, int((1 - t / skip) * skip * bars[x] + (t / skip) * skip * bars[x+1])))

		self.curve[0] = 0

	def process(self, command):

		if len(command) == 2:
			button, on = command

			if on and button == 'clear':
				self.reset()

			if on and button == 'print_to_clip':
				print(self.curve)
				print(self.bars)

		elif len(command) == 3:
			pad, on, velocity = command

			if pad == (0,7):
				self.tester = velocity

			elif on:
				for x in range(size):
					if x == pad[0]:
						self.bars[x] = (self.bars[x] + pad[1]) / 2

				self.fit()




	def print(self):
		self.display[:,:] = 0
		self.display[0,-1] = 9

		for x in range(size):
			bar = int(self.bars[x])
			top = self.colors[-int(len(self.colors) * (self.bars[x] - bar)) - 1]
			self.display[x, 0 : bar] = self.colors[0]
			self.display[x, bar] = top


		if self.tester:
			index = int(self.tester * (size / 2 ** 7) + 1/2)
			jndex = int(self.curve[self.tester] * (size / 2 ** 7) + 1/2)
			self.display[0 : index, 0] = 9
			self.display[index - 1, 0 : jndex] = 9



ports = mido.get_ioport_names()
surface = None
synth = None

if 'Launchpad Pro MK3 LPProMK3 MIDI' in ports:
	surface = 'Launchpad Pro MK3 LPProMK3 MIDI'
if 'Scarlett 18i8 USB' in ports:
	synth = 'Scarlett 18i8 USB'

with mido.open_ioport(surface) as ioport, \
	 mido.open_output(synth) as output:
	try:
		inst = Instrument(ioport, output)
		inst.display()
		for message in ioport:
			# if message.type == 'note_on': print(message)
			inst.process(message)
			inst.display()

			if not inst.running:
				break

	except KeyboardInterrupt:
		pass

	inst.reset()

'''
 - draw your own velocity curve
 - chord / voicing mode -- store pitch-class sets in banks (maybe also provide substitutions)
 	- rootless mode (roots do not sound)
 	- clockface mode (draw clock logo for chord storage)
 	- stored chords may be endowed list of permissible roots
 - 
'''

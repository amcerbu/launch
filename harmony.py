from time import sleep, time
from random import random

deep = False

n = 12

def argmin(a):
    return min(range(len(a)), key = lambda x : a[x])

def binary(n, b = 2):
	if n < b: return [n]
	return binary(n // b) + [n % b]

# generate all subsets of [0,1,...,k-1] for k in [0,...,n-1], filtered by cardinality
def power(n = n):
	subsets = [[] for i in range(n+1)]
	for i in range(2 ** n):
		b = binary(i)
		b = (n - len(b)) * [0] + b
		subsets[sum(b)].append([i for i, a in enumerate(b) if a])
	return subsets

# subsets[n][k] is all subsets of {0,...,n-1} of size k
subsets = [power(k) for k in range(n + 1)]

cutoff = n // 2 + 1 # for performance reasons we don't generate voicings of sets of size > 6
voicings = [None for k in range(n + 1)]

# generate all permutations of (0,...,n-1)
def permutations(n, voicings = voicings):
	if n == 0:
		voicings[n] = [()]
	
	if n >= len(voicings) or voicings[n] is None:
		voicings[n] = [p[:i] + (n-1,) + p[i:] for p in voicings[n-1] for i in range(n)]

	return voicings[n]

voicings = [permutations(k) for k in range(cutoff)]

# returns either an anchored or unanchored normal form of a pitch class set
def normal(pitches, transposition = False, n = n):
	if not pitches:
		return tuple(pitches)

	anchor = pitches[0]
	pitches = list(set((p - anchor) % n for p in pitches))
	pitches.sort()

	inversions = [pitches[i:] + pitches[:i] for i in range(len(pitches))]
	anchors = [inversion[0] for inversion in inversions]
	inversions = [[(p - inversion[0]) % n for p in inversion] for inversion in inversions]
	for inversion in inversions: inversion.sort()
	
	inversions = [tuple(inversion[::-1]) for inversion in inversions]
	index = argmin(inversions)
	anchor += anchors[index]
	anchor %= n
	if transposition:
		return inversions[index][::-1], anchor
	return inversions[index][::-1] 

def invert(pitches, n = n):
	return tuple(-p % n for p in pitches)

def complement(pitches, n = n):
	return tuple(p for p in range(n) if p not in pitches)

# generate all pitch class sets
def sets(subsets = subsets, n = n):
	accum = [set() for k in range(n + 1)]
	for k in range(n + 1):
		for choice in subsets[n][k]:
			accum[k].add(normal(choice))

		accum[k] = list(accum[k])
		accum[k].sort()

	return accum

encyclopedia = sets() # store pitch-class sets in accum

# dictionary of pitch-class set names (for classes of size <= 6) by normal form
forte = {() : '0-1',
		 (0,) : '1-1',
		 (0,1) : '2-1', (0,2) : '2-2', (0,3) : '2-3', (0,4) : '2-4', (0,5) : '2-5', (0,6) : '2-6',
		 (0,1,2) : '3-1', (0,1,3) : '3-2a', (0,2,3) : '3-2b', (0,1,4) : '3-3a', (0,3,4) : '3-3b',
		 	(0,1,5) : '3-4a', (0,4,5) : '3-4b', (0,1,6) : '3-5a', (0,5,6) : '3-5b', (0,2,4) : '3-6',
		 	(0,2,5) : '3-7a', (0,3,5) : '3-7b', (0,2,6) : '3-8a', (0,4,6) : '3-8b', (0,2,7) : '3-9',
		 	(0,3,6) : '3-10', (0,3,7) : '3-11a', (0,4,7) : '3-11b', (0,4,8) : '3-12',
		 (0,1,2,3) : '4-1', (0,1,2,4) : '4-2a', (0,2,3,4) : '4-2b', (0,1,3,4) : '4-3', (0,1,2,5) : '4-4a',
		 	(0,3,4,5) : '4-4b', (0,1,2,6) : '4-5a', (0,4,5,6) : '4-5b', (0,1,2,7) : '4-6', (0,1,4,5) : '4-7',
		 	(0,1,5,6) : '4-8', (0,1,6,7) : '4-9', (0,2,3,5) : '4-10', (0,1,3,5) : '4-11a', (0,2,4,5) : '4-11b',
		 	(0,2,3,6) : '4-12a', (0,3,4,6) : '4-12b', (0,1,3,6) : '4-13a', (0,3,5,6) : '4-13b', (0,2,3,7) : '4-14a',
		 	(0,4,5,7) : '4-14b', (0,1,4,6) : '4-z15a', (0,2,5,6) : '4-z15b', (0,1,5,7) : '4-16a', (0,2,6,7) : '4-16b',
		 	(0,3,4,7) : '4-17', (0,1,4,7) : '4-18a', (0,3,6,7) : '4-18b', (0,1,4,8) : '4-19a', (0,3,4,8) : '4-19b',
		 	(0,1,5,8) : '4-20', (0,2,4,6) : '4-21', (0,2,4,7) : '4-22a', (0,3,5,7) : '4-22b', (0,2,5,7) : '4-23',
		 	(0,2,4,8) : '4-24', (0,2,6,8) : '4-25', (0,3,5,8) : '4-26', (0,2,5,8) : '4-27a', (0,3,6,8) : '4-27b',
		 	(0,3,6,9) : '4-28', (0,1,3,7) : '4-z29a', (0,4,6,7) : '4-z29b',
		 (0,1,2,3,4) : '5-1', (0,1,2,3,5) : '5-2a', (0,2,3,4,5) : '5-2b', (0,1,2,4,5) : '5-3a', (0,1,3,4,5) : '5-3b',
		 	(0,1,2,3,6) : '5-4a', (0,3,4,5,6) : '5-4b', (0,1,2,3,7) : '5-5a', (0,4,5,6,7) : '5-5b', (0,1,2,5,6) : '5-6a',
		 	(0,1,4,5,6) : '5-6b', (0,1,2,6,7) : '5-7a', (0,1,5,6,7) : '5-7b', (0,2,3,4,6) : '5-8', (0,1,2,4,6) : '5-9a',
		 	(0,2,4,5,6) : '5-9b', (0,1,3,4,6) : '5-10a', (0,2,3,5,6) : '5-10b', (0,2,3,4,7) : '5-11a', (0,3,4,5,7) : '5-11b',
		 	(0,1,3,5,6) : '5-z12', (0,1,2,4,8) : '5-13a', (0,2,3,4,8) : '5-13b', (0,1,2,5,7) : '5-14a', (0,2,5,6,7) : '5-14b',
		 	(0,1,2,6,8) : '5-15', (0,1,3,4,7) : '5-16a', (0,3,4,6,7) : '5-16b', (0,1,3,4,8) : '5-z17', (0,1,4,5,7) : '5-z18a',
		 	(0,2,3,6,7) : '5-z18b', (0,1,3,6,7) : '5-19a', (0,1,4,6,7) : '5-19b', (0,1,5,6,8) : '5-20a', (0,2,3,7,8) : '5-20b',
		 	(0,1,4,5,8) : '5-21a', (0,3,4,7,8) : '5-21b', (0,1,4,7,8) : '5-22', (0,2,3,5,7) : '5-23a', (0,2,4,5,7) : '5-23b',
		 	(0,1,3,5,7) : '5-24a', (0,2,4,6,7) : '5-24b', (0,2,3,5,8) : '5-25a', (0,3,5,6,8) : '5-25b', (0,2,4,5,8) : '5-26a',
		 	(0,3,4,6,8) : '5-26b', (0,1,3,5,8) : '5-27a', (0,3,5,7,8) : '5-27b', (0,2,3,6,8) : '5-28a', (0,2,5,6,8) : '5-28b',
		 	(0,1,3,6,8) : '5-29a', (0,2,5,7,8) : '5-29b', (0,1,4,6,8) : '5-30a', (0,2,4,7,8) : '5-30b', (0,1,3,6,9) : '5-31a',
		 	(0,2,3,6,9) : '5-31b', (0,1,4,6,9) : '5-32a', (0,2,5,6,9) : '5-32b', (0,2,4,6,8) : '5-33', (0,2,4,6,9) : '5-34',
		 	(0,2,4,7,9) : '5-35', (0,1,2,4,7) : '5-z36a', (0,3,5,6,7) : '5-z36b', (0,3,4,5,8) : '5-z37', (0,1,2,5,8) : '5-z38a',
		 	(0,3,6,7,8) : '5-z38b',
		 (0,1,2,3,4,5) : '6-1', (0,1,2,3,4,6) : '6-2a', (0,2,3,4,5,6) : '6-2b', (0,1,2,3,5,6) : '6-z3a', (0,1,3,4,5,6) : '6-z3b',
		 	(0,1,2,4,5,6) : '6-z4', (0,1,2,3,6,7) : '6-5a', (0,1,4,5,6,7) : '6-5b', (0,1,2,5,6,7) : '6-z6', (0,1,2,6,7,8) : '6-7',
		 	(0,2,3,4,5,7) : '6-8', (0,1,2,3,5,7) : '6-9a', (0,2,4,5,6,7) : '6-9b', (0,1,3,4,5,7) : '6-10a', (0,2,3,4,6,7) : '6-10b',
		 	(0,1,2,4,5,7) : '6-z11a', (0,2,3,5,6,7) : '6-z11b', (0,1,2,4,6,7) : '6-z12a', (0,1,3,5,6,7) : '6-z12b', (0,1,3,4,6,7) : '6-z13',
		 	(0,1,3,4,5,8) : '6-14a', (0,3,4,5,7,8) : '6-14b', (0,1,2,4,5,8) : '6-15a', (0,3,4,6,7,8) : '6-15b', (0,1,4,5,6,8) : '6-16a',
		 	(0,2,3,4,7,8) : '6-16b', (0,1,2,4,7,8) : '6-z17a', (0,1,4,6,7,8) : '6-z17b', (0,1,2,5,7,8) : '6-18a', (0,1,3,6,7,8) : '6-18b',
		 	(0,1,3,4,7,8) : '6-z19a', (0,1,4,5,7,8) : '6-z19b', (0,1,4,5,8,9) : '6-20', (0,2,3,4,6,8) : '6-21a', (0,2,4,5,6,8) : '6-21b',
		 	(0,1,2,4,6,8) : '6-22a', (0,2,4,6,7,8) : '6-22b', (0,2,3,5,6,8) : '6-z23', (0,1,3,4,6,8) : '6-z24a', (0,2,4,5,7,8) : '6-z24b',
		 	(0,1,3,5,6,8) : '6-z25a', (0,2,3,5,7,8) : '6-z25b', (0,1,3,5,7,8) : '6-z26', (0,1,3,4,6,9) : '6-27a', (0,2,3,5,6,9) : '6-27b',
		 	(0,1,3,5,6,9) : '6-z28', (0,2,3,6,7,9) : '6-z29', (0,1,3,6,7,9) : '6-30a', (0,2,3,6,8,9) : '6-30b', (0,1,4,5,7,9) : '6-31a',
		 	(0,2,4,5,8,9) : '6-31b', (0,2,4,5,7,9) : '6-32', (0,2,3,5,7,9) : '6-33a', (0,2,4,6,7,9) : '6-33b', (0,1,3,5,7,9) : '6-34a',
		 	(0,2,4,6,8,9) : '6-34b', (0,2,4,6,8,10) : '6-35', (0,3,4,5,6,7) : '6-z36b', (0,1,2,3,4,7) : '6-z36a', (0,1,2,3,4,8) : '6-z37',
		 	(0,1,2,3,7,8) : '6-z38', (0,3,4,5,6,8) : '6-z39b', (0,2,3,4,5,8) : '6-z39a', (0,3,5,6,7,8) : '6-z40b', (0,1,2,3,5,8) : '6-z40a',
		 	(0,2,5,6,7,8) : '6-z41b', (0,1,2,3,6,8) : '6-z41a', (0,1,2,3,6,9) : '6-z42', (0,2,3,6,7,8) : '6-z43b', (0,1,2,5,6,8) : '6-z43a',
		 	(0,1,4,5,6,9) : '6-z44b', (0,1,2,5,6,9) : '6-z44a', (0,2,3,4,6,9) : '6-z45', (0,2,4,5,6,9) : '6-z46b', (0,1,2,4,6,9) : '6-z46a',
		 	(0,2,3,4,7,9) : '6-z47b', (0,1,2,4,7,9) : '6-z47a', (0,1,2,5,7,9) : '6-z48', (0,1,3,4,7,9) : '6-z49', (0,1,4,6,7,9) : '6-z50'}

# completes the forte table
def complete(forte = forte, n = n):
	temp = {}
	for prime in forte:
		if len(prime) < n // 2:
			assert prime == normal(prime)
			comp = normal(complement(prime)) # complement
			inv = normal(invert(comp)) # inversion of complement
			code = forte[prime][1:] # forte code for original, without leading number
			if code[-1] in 'ab':
				if code[-1] == 'a': # sanity check: naming convention is intact
					assert prime[::-1] < normal(invert(prime))[::-1]
				code = code[:-1] # strip trailing a or b
			else:
				assert prime == normal(invert(prime))
			inversion = '' if comp == inv else 'a' if comp[::-1] < inv[::-1] else 'b'
			temp[comp] = str(len(comp)) + code + inversion
	forte.update(temp)

complete()

# build reverse lookup-table for retrieving pitch-class sets by name (with optional omission of z or a/b)
lookup = {v : k for k, v in forte.items()}
lookup.update({k[:-1] : v for k, v in lookup.items() if k[-1] == 'a'})
lookup.update({k[:2] + k[3:] : v for k, v in lookup.items() if 'z' in k})

# counts the occurrences of sub inside sup
def contains(sub, sup, normalized = False, subsets = subsets):
	if not normalized: sub = normal(sub)
	n = len(sup)
	k = len(sub)
	count = 0
	choice = subsets[n][k] if 0 <= k <= n else []
	for I in choice:
		count += int(sub == normal([sup[i] for i in I]))
	return count

# computes a "generalized interval vector"
def vector(pitches, k, name = False, symmetrize = False, forte = forte):
	primes = [prime for prime in forte if len(prime) == k]
	if isinstance(pitches, str): pitches = lookup[pitches]

	if symmetrize:
		for i, prime in enumerate(primes):
			inverse = normal(invert(prime))
			if inverse != prime and inverse in primes:
				primes[i] = None
		primes = [prime for prime in primes if prime is not None]

	temp = {}
	for prime in primes:
		key = forte[prime] if name else prime
		if symmetrize:
			inverse = normal(invert(prime))
			if inverse != prime:
				if name: key = key[:-1]
				count = contains(inverse, pitches, normalized = True)
				temp[key] = count if key not in temp else temp[key] + count

		count = contains(prime, pitches, normalized = True)
		temp[key] = count if key not in temp else temp[key] + count
	return temp

# displays a clockface of pitches
def clock(pitches, n = n):
	assert n == 12
	if isinstance(pitches, str): pitches = lookup[pitches]

	template = '    3  2\n 4        1\n5          0\n6          E\n 7        T\n    8  9'
	breakup = list(template)
	key = {i : breakup.index(j) for i, j in enumerate('0123456789TE')}
	
	pitches = normal(pitches)
	offset = 0 if not pitches else int((pitches[-1] - 4) // 2)
	pitches = [(p - offset) % n for p in pitches]

	for p in key:
		if p not in pitches:
			breakup[key[p]] = '·'
		else:
			breakup[key[p]] = '0123456789TE'[(p + offset) % n]

	print(''.join(breakup))

# returns list of pitch class sets containing all k-ads
def allad(k, symmetrize = False, forte = forte, lookup = lookup):
	A = []
	for p in forte:
		v = vector(p, k, name = True, symmetrize = symmetrize)
		v = [val for key, val in v.items()]
		if 0 not in v:
			A.append(p)
	A.sort()
	return A

def clean(key, symmetrize):
	name = type(key) == str
	if symmetrize:
		if name:
			if key[-1] in 'ab': key = key[:-1]
		else: 
			key = min(key[::-1], normal(invert(key))[::-1])[::-1]
	return key

# build the poset of pitch-class sets, with edges weighted by inclusion
def hierarchy(name = False, symmetrize = False, encyclopedia = encyclopedia, forte = forte):
	poset = {}
	for page in encyclopedia[::-1]:
		if symmetrize:
			exclude = []
			for i, prime in enumerate(page):
				inverse = normal(invert(prime))
				if inverse != prime and inverse not in [page[i] for i in exclude]:
					exclude.append(i)
			page = [prime for i, prime in enumerate(page) if i not in exclude]

		for pitches in page:
			children = vector(pitches, len(pitches) - 1, name, symmetrize)
			children = [key for key, val in children.items() if val]

			key = forte[pitches] if name else pitches
			key = clean(key, symmetrize)

			poset[key] = children

	return poset

if deep:
	poset = hierarchy(name = True, symmetrize = True)
	chiral = hierarchy(name = True, symmetrize = False)
	family = hierarchy(name = False, symmetrize = False)

	# retrieve all descendants of a give set of nodes
	def grow(A, poset = poset):
		children = set()
		for a in A:
			children.update(poset[a])
		return children

	# find largest-cardinality intersections between two pitch-class sets
	def intersect(A, B, poset = poset):
		name = type(next(iter(poset.keys()))) == str
		assert type(A) == type(B) == (str if name else tuple)

		childrenA = set()
		childrenB = set()

		a = [A]
		while a: childrenA.update(a); a = grow(a, poset)

		b = [B]
		while b: childrenB.update(b); b = grow(b, poset)

		overlap = set.intersection(childrenA, childrenB)
		proximity = max((len(lookup[o] if name else o) for o in overlap), default = 0)
		return set(o for o in overlap if len(lookup[o] if name else o) == proximity)

# voice a pitch-class set according to a permutation
def voice(pitches, permutation, n = n, ground = False):
	pitches = [pitches[p] for p in permutation]
	for i in range(1, len(pitches)):
		if pitches[i] < pitches[i-1]:
			pitches[i:] = [p + n for p in pitches[i:]]
	return tuple(p - (pitches[0] if ground else 0) for p in pitches)

# return all voicings stratified by breadth
def stratify(pitches, n = n):
	A = [voice(pitches, pi) for pi in voicings[len(pitches)]]
	B = [[] for p in pitches]
	for i in range(len(pitches)):
		B[i] = [a for a in A if a[0] == pitches[i]]

	C = [[] for p in pitches]
	for i in range(len(pitches)):
		C[i] = [[a for a in B[i] if max(a[i] - a[i-1] for i in range(1, len(a))) == j] for j in range(n)]
		C[i] = [x for x in C[i] if x]

	return C

# return all voicings sorted by root
def root(pitches, n = n):
	A = [voice(pitches, pi) for pi in voicings[len(pitches)]]
	return [[a for a in A if max(a[i] - a[i-1] for i in range(1, len(a))) == j] for j in range(n)]

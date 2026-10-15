package engine

import "math/rand/v2"

// bag is a 7-bag randomizer: every run of 7 pieces contains each kind exactly once.
type bag struct {
	rng   *rand.Rand
	kinds [NumKinds]Kind
	pos   int
}

func newBag(seed uint64) *bag {
	b := &bag{rng: rand.New(rand.NewPCG(seed, 0))}
	b.pos = len(b.kinds)
	return b
}

func (b *bag) pop() Kind {
	if b.pos == len(b.kinds) {
		for i := range b.kinds {
			b.kinds[i] = Kind(i)
		}
		b.rng.Shuffle(len(b.kinds), func(i, j int) { b.kinds[i], b.kinds[j] = b.kinds[j], b.kinds[i] })
		b.pos = 0
	}
	k := b.kinds[b.pos]
	b.pos++
	return k
}

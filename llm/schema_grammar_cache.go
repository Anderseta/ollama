package llm

import (
	"container/list"
	"sync"
)

const (
	schemaGrammarsSize     = 64
	schemaGrammarsMaxBytes = 8 << 20
)

// The native schema converter is fixed for the process. Bound both retained
// entries and their string payloads; large conversions still work uncached.
var schemaGrammars = newSchemaGrammarCache(schemaGrammarsSize, schemaGrammarsMaxBytes)

type schemaGrammar struct {
	schema, grammar string
}

type schemaGrammarCache struct {
	mu         sync.Mutex
	entries    map[string]*list.Element
	order      list.List
	bytes      int
	maxEntries int
	maxBytes   int
}

func newSchemaGrammarCache(maxEntries, maxBytes int) *schemaGrammarCache {
	return &schemaGrammarCache{
		entries: make(map[string]*list.Element), maxEntries: maxEntries, maxBytes: maxBytes,
	}
}

func (c *schemaGrammarCache) get(schema string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.entries[schema]; ok {
		c.order.MoveToFront(e)
		return e.Value.(*schemaGrammar).grammar, true
	}
	return "", false
}

func (c *schemaGrammarCache) put(schema, grammar string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.entries[schema]; ok {
		c.order.MoveToFront(e)
		return
	}
	if c.maxEntries <= 0 || len(schema) > c.maxBytes || len(grammar) > c.maxBytes-len(schema) {
		return
	}
	cost := len(schema) + len(grammar)
	for len(c.entries) >= c.maxEntries || c.bytes > c.maxBytes-cost {
		oldest := c.order.Back()
		item := oldest.Value.(*schemaGrammar)
		c.bytes -= len(item.schema) + len(item.grammar)
		delete(c.entries, item.schema)
		c.order.Remove(oldest)
	}
	c.entries[schema] = c.order.PushFront(&schemaGrammar{schema: schema, grammar: grammar})
	c.bytes += cost
}

func (c *schemaGrammarCache) getOrConvert(schema string, convert func() (string, error)) (string, error) {
	if grammar, ok := c.get(schema); ok {
		return grammar, nil
	}
	grammar, err := convert()
	if err != nil {
		return "", err
	}
	c.put(schema, grammar)
	return grammar, nil
}

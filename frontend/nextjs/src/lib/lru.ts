// Small insertion-ordered LRU cache.
//
// JS `Map` preserves insertion order, so `keys().next()` is always the
// least-recently-used entry. Re-inserting a key on access moves it to the end,
// marking it as freshly used. When the cache exceeds its capacity the oldest
// (front) entry is evicted.
//
// Used to keep several recently-viewed responses "hot" so flipping back and
// forth between them (e.g. comparing timestamps in Explain mode) is instant
// instead of re-hitting the backend.
export class LRUCache<V> {
  private map = new Map<string, V>();

  constructor(private capacity: number) {}

  get(key: string): V | undefined {
    const v = this.map.get(key);
    if (v !== undefined) {
      // Mark as most-recently-used by re-inserting at the end.
      this.map.delete(key);
      this.map.set(key, v);
    }
    return v;
  }

  has(key: string): boolean {
    return this.map.has(key);
  }

  set(key: string, value: V): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value;
      if (oldest !== undefined) this.map.delete(oldest);
    }
  }

  clear(): void {
    this.map.clear();
  }
}

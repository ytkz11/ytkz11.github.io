'use strict';
const searchForm = document.querySelector('.search-form');
if (searchForm) {
  const input = document.querySelector('#search-input');
  const status = document.querySelector('#search-status');
  const results = document.querySelector('#search-results');
  let indexPromise;
  let request = 0;
  async function search() {
    const current = ++request;
    const query = input.value.trim();
    const url = new URL(location.href);
    query ? url.searchParams.set('q', query) : url.searchParams.delete('q');
    history.replaceState(null, '', url);
    results.replaceChildren();
    if (!query) { status.textContent = '输入关键词开始搜索。'; return; }
    status.textContent = '正在搜索…';
    try {
      if (!indexPromise) indexPromise = fetch(results.dataset.index).then(r => { if (!r.ok) throw new Error('index'); return r.json(); }).catch(e => { indexPromise = null; throw e; });
      const posts = await indexPromise;
      if (current !== request) return;
      const words = query.toLocaleLowerCase().split(/\s+/);
      const matches = posts.filter(p => words.every(word => [p.title, ...p.tags, ...p.categories].join(' ').toLocaleLowerCase().includes(word)));
      status.textContent = matches.length ? `找到 ${matches.length} 篇文章` : '没有找到相关文章，试试更短的关键词。';
      for (const post of matches) {
        const link = document.createElement('a'); link.className = 'search-result'; link.href = post.url;
        const title = document.createElement('h2'); title.textContent = post.title;
        const meta = document.createElement('p'); meta.textContent = [post.date, ...post.categories].join(' · ');
        link.append(title, meta); results.append(link);
      }
    } catch { if (current === request) status.textContent = '搜索索引加载失败，请稍后重新搜索。'; }
  }
  searchForm.addEventListener('submit', e => { e.preventDefault(); search(); });
  let timer;
  input.addEventListener('input', () => { ++request; clearTimeout(timer); timer = setTimeout(search, 180); });
  input.value = new URLSearchParams(location.search).get('q') || '';
  if (input.value) search();
}

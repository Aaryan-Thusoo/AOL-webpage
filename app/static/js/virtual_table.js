// app/static/js/virtual_table.js
export class VirtualTable {
  constructor(container, columns, fetchPage, opts = {}) {
    this.container = container;
    this.columns = columns;
    this.fetchPage = fetchPage;
    this.rowHeight = opts.rowHeight ?? 32;
    this.pageSize = opts.pageSize ?? 200;
    this.buffer = opts.buffer ?? 8;
    this.hasMore = true;
    this.cache = new Map();
    this.maxLoaded = 0;
    this._buildDOM();
    this._attach();
    this._ensureVisibleData().catch(console.error);
  }

  _buildDOM() {
    this.container.style.position = "relative";
    this.container.style.overflow = "auto";

    this.header = document.createElement("div");
    this.header.style.position = "sticky";
    this.header.style.top = "0";
    this.header.style.zIndex = "2";
    this.header.style.background = "#f3f4f6";
    this.header.style.borderBottom = "1px solid #e5e7eb";
    this.header.style.display = "grid";
    this.header.style.gridTemplateColumns = `repeat(${this.columns.length}, minmax(120px, 1fr))`;
    this.header.style.fontWeight = "600";
    this.header.style.padding = "6px 8px";

    for (const c of this.columns) {
      const cell = document.createElement("div");
      Object.assign(cell.style, {
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis"
      });
      cell.textContent = c;
      this.header.appendChild(cell);
    }

    this.spacer = document.createElement("div");
    this.spacer.style.height = "0px";

    this.items = document.createElement("div");
    this.items.style.position = "absolute";
    this.items.style.left = "0";
    this.items.style.right = "0";
    this.items.style.display = "grid";
    this.items.style.gridTemplateColumns = `repeat(${this.columns.length}, minmax(120px, 1fr))`;

    this.container.innerHTML = "";
    this.container.appendChild(this.header);
    this.container.appendChild(this.spacer);
    this.container.appendChild(this.items);
  }

  _attach() {
    this._onScroll = () => this._ensureVisibleData().catch(console.error);
    this.container.addEventListener("scroll", this._onScroll, { passive: true });
    new ResizeObserver(() => this._render()).observe(this.container);
  }

  async _ensureVisibleData() {
    const viewTop = this.container.scrollTop;
    const viewHeight = this.container.clientHeight;
    const from = Math.max(0, Math.floor(viewTop / this.rowHeight) - this.buffer);
    const to = Math.floor((viewTop + viewHeight) / this.rowHeight) + this.buffer;

    let offset = Math.floor(from / this.pageSize) * this.pageSize;

    while (offset <= to && this.hasMore) {
      if (!this.cache.has(offset)) {
        const result = await this.fetchPage(offset, this.pageSize);
        const rows = result.rows || [];
        const hasMore = result.hasMore !== undefined ? result.hasMore : result.has_more;

        this.cache.set(offset, rows);
        this.maxLoaded = Math.max(this.maxLoaded, offset + rows.length);
        this.hasMore = hasMore !== false;
      }
      offset += this.pageSize;
    }

    const maxRows = this.hasMore ? Math.max(this.maxLoaded, to + this.pageSize) : this.maxLoaded;
    this.spacer.style.height = `${maxRows * this.rowHeight + this.header.offsetHeight}px`;
    this._render();
  }

  _render() {
    const viewTop = this.container.scrollTop;
    const viewHeight = this.container.clientHeight;
    const first = Math.max(0, Math.floor(viewTop / this.rowHeight) - this.buffer);
    const last = Math.floor((viewTop + viewHeight) / this.rowHeight) + this.buffer;

    const visible = [];
    for (let r = first; r <= last; r++) {
      const pageOffset = Math.floor(r / this.pageSize) * this.pageSize;
      const page = this.cache.get(pageOffset);
      if (!page) continue;

      const idx = r - pageOffset;
      if (idx >= 0 && idx < page.length) visible.push(page[idx]);
    }

    this.items.innerHTML = "";
    const startY = first * this.rowHeight + this.header.offsetHeight;
    this.items.style.transform = `translateY(${startY}px)`;

    for (const row of visible) {
      for (let c = 0; c < this.columns.length; c++) {
        const cell = document.createElement("div");
        const v = row[c];
        cell.textContent = v == null ? "" : String(v);
        Object.assign(cell.style, {
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
          borderBottom: "1px solid #eee",
          padding: "6px 8px",
          height: `${this.rowHeight}px`,
          lineHeight: `${this.rowHeight - 12}px`,
        });
        this.items.appendChild(cell);
      }
    }
  }

  destroy() {
    this.container.removeEventListener("scroll", this._onScroll);
    this.container.innerHTML = "";
    this.cache.clear();
  }
}
from playwright.sync_api import sync_playwright, Page, Locator
from playwright_stealth import Stealth
from urllib.parse import quote
from typing import Callable, Optional
import os
import time
import re
import json
import requests

BASE_URL = "https://www.google.com/search?q="

# ---------------------------------------------------------------------------
# Schema unificado: propriedades + ações possíveis por role
# ---------------------------------------------------------------------------

ROLE_SCHEMA = {
    "document": {"properties": ["label"], "actions": ["read"]},
    "main": {"properties": ["label"], "actions": ["read"]},
    "navigation": {"properties": ["label"], "actions": ["read"]},
    "banner": {"properties": ["label"], "actions": ["read"]},
    "contentinfo": {"properties": ["label"], "actions": ["read"]},
    "complementary": {"properties": ["label"], "actions": ["read"]},
    "region": {"properties": ["label"], "actions": ["read"]},
    "search": {"properties": ["label"], "actions": ["read"]},
    "form": {"properties": ["label"], "actions": ["read", "submit"]},
    "menu": {"properties": ["label", "options", "selected"], "actions": ["read", "expand", "collapse"]},
    "menubar": {"properties": ["label"], "actions": ["read"]},
    "tablist": {"properties": ["label"], "actions": ["read"]},
    "list": {"properties": ["label"], "actions": ["read"]},
    "grid": {"properties": ["label"], "actions": ["read"]},
    "table": {"properties": ["label"], "actions": ["read"]},
    "tree": {"properties": ["label"], "actions": ["read"]},

    "heading": {"properties": ["text", "level"], "actions": ["read"]},
    "paragraph": {"properties": ["text"], "actions": ["read"]},
    "note": {"properties": ["text"], "actions": ["read"]},
    "caption": {"properties": ["text"], "actions": ["read"]},

    "button": {"properties": ["text", "disabled", "pressed"], "actions": ["read", "click", "press"]},
    "link": {"properties": ["text", "url", "disabled"], "actions": ["click", "press"]},
    "textbox": {"properties": ["value", "placeholder", "disabled", "readonly"], "actions": ["read", "type", "clear", "press"]},
    "searchbox": {"properties": ["value", "placeholder", "disabled", "readonly"], "actions": ["read", "type", "clear", "submit", "press"]},
    "combobox": {"properties": ["value", "options", "expanded", "disabled"], "actions": ["read", "type", "select", "expand", "collapse", "press"]},
    "listbox": {"properties": ["options", "selected"], "actions": ["read", "select", "press"]},
    "option": {"properties": ["text", "selected", "disabled"], "actions": ["read", "select", "press"]},
    "menuitem": {"properties": ["text", "selected", "disabled"], "actions": ["read", "click", "press"]},
    "treeitem": {"properties": ["text", "selected", "disabled", "expanded"], "actions": ["read", "click", "expand", "collapse", "press"]},
    "checkbox": {"properties": ["text", "checked"], "actions": ["read", "toggle", "press"]},
    "switch": {"properties": ["text", "checked"], "actions": ["read", "toggle", "press"]},
    "radio": {"properties": ["text", "checked", "group"], "actions": ["read", "select", "press"]},
    "slider": {"properties": ["value", "min", "max"], "actions": ["read", "set_value", "press"]},
    "progressbar": {"properties": ["value", "min", "max"], "actions": ["read"]},
    "meter": {"properties": ["value", "min", "max"], "actions": ["read"]},

    "tab": {"properties": ["text", "selected", "panel_id"], "actions": ["read", "click", "press"]},
    "tabpanel": {"properties": ["label"], "actions": ["read"]},

    "listitem": {"properties": ["text", "position"], "actions": ["read"]},
    "gridcell": {"properties": ["text", "row", "column"], "actions": ["read", "click", "press"]},
    "columnheader": {"properties": ["text", "row", "column"], "actions": ["read", "click"]},
    "rowheader": {"properties": ["text", "row", "column"], "actions": ["read"]},
    "row": {"properties": ["position"], "actions": ["read", "click"]},

    "dialog": {"properties": ["label", "modal"], "actions": ["read", "close"]},
    "alertdialog": {"properties": ["label", "modal"], "actions": ["read", "close"]},
    "alert": {"properties": ["text", "urgency"], "actions": ["read"]},
    "status": {"properties": ["text", "urgency"], "actions": ["read"]},

    "img": {"properties": ["alt", "url"], "actions": ["read"]},
    "figure": {"properties": ["label"], "actions": ["read"]},
}


def _get_value(loc: Locator) -> Optional[str]:
    try:
        return loc.input_value()
    except Exception:
        return loc.get_attribute("value")


def _get_checked(loc: Locator):
    try:
        return loc.is_checked()
    except Exception:
        return loc.get_attribute("aria-checked")


PROPERTY_EXTRACTORS: dict[str, Callable[[Locator], Optional[str]]] = {
    "text": lambda loc: loc.inner_text(),
    "label": lambda loc: loc.get_attribute("aria-label") or loc.inner_text(),
    "value": _get_value,
    "url": lambda loc: loc.get_attribute("href") or loc.get_attribute("src"),
    "alt": lambda loc: loc.get_attribute("alt"),
    "level": lambda loc: loc.get_attribute("aria-level"),
    "disabled": lambda loc: loc.is_disabled(),
    "readonly": lambda loc: loc.get_attribute("readonly") is not None,
    "placeholder": lambda loc: loc.get_attribute("placeholder"),
    "checked": _get_checked,
    "pressed": lambda loc: loc.get_attribute("aria-pressed"),
    "expanded": lambda loc: loc.get_attribute("aria-expanded"),
    "selected": lambda loc: loc.get_attribute("aria-selected"),
    "min": lambda loc: loc.get_attribute("aria-valuemin"),
    "max": lambda loc: loc.get_attribute("aria-valuemax"),
    "group": lambda loc: loc.get_attribute("name"),
    "position": lambda loc: loc.get_attribute("aria-posinset"),
    "row": lambda loc: loc.get_attribute("aria-rowindex"),
    "column": lambda loc: loc.get_attribute("aria-colindex"),
    "panel_id": lambda loc: loc.get_attribute("aria-controls"),
    "modal": lambda loc: loc.get_attribute("aria-modal"),
    "urgency": lambda loc: loc.get_attribute("aria-live"),
    "options": lambda loc: None,
}


def clean_text(text) -> Optional[str]:
    """Remove espaços não-quebráveis, normaliza quebras de linha e espaços redundantes."""
    if not text:
        return None
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    return text or None


def is_empty_item(item_info: dict) -> bool:
    """Considera 'vazio' um item onde toda propriedade relevante é None/False/string vazia."""
    for key, value in item_info.items():
        if key in ("role", "actions", "disabled", "checked", "pressed", "expanded", "selected", "readonly", "modal"):
            continue  # estados/metadados, não indicam conteúdo
        if value:
            return False
    return True


# ---------------------------------------------------------------------------
# Extração genérica de elementos da página (com properties + actions)
# ---------------------------------------------------------------------------

SKIP_ROLES = {"document", "main"}  # geram blob gigante redundante com o resto

# Atributo gravado no próprio elemento do DOM durante a extração, contendo o
# id numérico do item. É por ele que execute_item_action() re-localiza o
# elemento depois, sem precisar guardar ElementHandle (que fica inválido após
# qualquer re-render) nem depender de seletores frágeis.
ITEM_ID_ATTR = "data-ava-id"

GENERIC_EXTRACTION_JS = r"""
(idAttr) => {
    const MAX_TEXT = 160;
    const MAX_VALUE = 200;
    const MAX_META = 3;
    const MAX_OPTIONS = 25;
    const TOTAL_CAP = 600;
    const SEP_RE = /\s*(?:•|·|\|)\s*/;

    // Remove ids de uma extração anterior antes de atribuir os novos.
    document.querySelectorAll(`[${idAttr}]`).forEach(el => el.removeAttribute(idAttr));
    let nextId = 1;

    const isVisible = (el) => {
        if (!el.isConnected) return false;
        const style = getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) === 0) return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    };

    const cleanText = (t) => (t || '').replace(/\s+/g, ' ').trim();
    const aria = (el, name) => el.getAttribute(name);
    const isDisabled = (el) => el.disabled === true || aria(el, 'aria-disabled') === 'true';

    // Rótulo curto. allowInnerText=false impede blobs gigantes em containers
    // (menu, nav, tabpanel...): nesses só vale aria-label/heading/label.
    function shortLabel(el, allowInnerText = true) {
        const ariaLabel = aria(el, 'aria-label');
        if (ariaLabel) return { text: cleanText(ariaLabel), meta: null };

        if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') {
            const placeholder = el.getAttribute('placeholder');
            if (placeholder) return { text: cleanText(placeholder), meta: null };
        }

        if (el.labels && el.labels.length) {
            return { text: cleanText(el.labels[0].innerText), meta: null };
        }

        if (el.tagName === 'IMG') {
            const alt = el.getAttribute('alt');
            if (alt) return { text: cleanText(alt), meta: null };
        }

        const heading = el.querySelector('h1,h2,h3,h4,h5,h6,[role="heading"]');
        if (heading) {
            const title = cleanText(heading.innerText);
            const rest = allowInnerText ? cleanText(el.innerText).replace(title, '').trim() : '';
            const meta = rest ? rest.split(SEP_RE).filter(Boolean).slice(0, MAX_META).join(' • ') : null;
            return { text: title, meta };
        }

        if (!allowInnerText) return { text: null, meta: null };

        const full = cleanText(el.innerText || el.value || '');
        if (full.length <= MAX_TEXT) return { text: full, meta: null };

        const parts = full.split(SEP_RE).filter(Boolean);
        return { text: parts[0].slice(0, MAX_TEXT), meta: parts.slice(1, 4).join(' • ') || null };
    }

    const results = [];
    const seen = new Set();

    // fallbackText: string|function usada quando a heurística não acha nome.
    function push(role, el, extra, fallbackText, allowInnerText) {
        extra = extra || {};
        if (!isVisible(el) || results.length >= TOTAL_CAP) return;

        const { text, meta } = shortLabel(el, allowInnerText !== false);
        const fb = typeof fallbackText === 'function' ? fallbackText() : fallbackText;
        const finalText = text || cleanText(fb || '');
        if (!finalText) return;

        const key = role + '|' + finalText + '|' + (el.href || '');
        if (seen.has(key)) return;
        seen.add(key);

        const id = nextId++;
        el.setAttribute(idAttr, String(id));

        const item = Object.assign({ id, role, text: finalText }, extra);
        if (meta) item.meta = meta;
        if (isDisabled(el)) item.disabled = true;  // agente VÊ que existe, mas não pode agir
        results.push(item);
    }

    // Nomes default de landmarks, com contador: dois <nav> sem aria-label
    // não podem colidir no dedupe como "navigation" duplicado.
    const defaultNameCount = new Map();
    const defaultName = (base) => () => {
        const n = (defaultNameCount.get(base) || 0) + 1;
        defaultNameCount.set(base, n);
        return n === 1 ? base : `${base} #${n}`;
    };

    /* -------------- Fase 1: elementos interativos -------------- */

    document.querySelectorAll('a[href]').forEach(el => push('link', el, { url: el.href }));

    document.querySelectorAll(
        'button, [role="button"], input[type="submit"], input[type="button"], input[type="reset"], input[type="image"]'
    ).forEach(el => push('button', el, { pressed: aria(el, 'aria-pressed') }));

    document.querySelectorAll(
        'input:not([type="hidden"]):not([type="submit"]):not([type="button"]):not([type="reset"]):not([type="image"])' +
        ':not([type="checkbox"]):not([type="radio"]):not([type="range"]):not([type="file"]):not([type="search"]):not([list]), textarea'
    ).forEach(el => push('textbox', el, {
        value: (el.value || '').slice(0, MAX_VALUE) || null,
        placeholder: el.getAttribute('placeholder'),
        readonly: el.readOnly ? true : null,
    }));

    document.querySelectorAll('input[type="search"]').forEach(el => push('searchbox', el, {
        value: (el.value || '').slice(0, MAX_VALUE) || null,
        placeholder: el.getAttribute('placeholder'),
        readonly: el.readOnly ? true : null,
    }));

    document.querySelectorAll('input[type="checkbox"]:not([role="switch"]), [role="checkbox"]').forEach(el =>
        push('checkbox', el, { checked: el instanceof HTMLInputElement ? el.checked : aria(el, 'aria-checked') === 'true' }));

    document.querySelectorAll('[role="switch"]').forEach(el =>
        push('switch', el, { checked: el instanceof HTMLInputElement ? el.checked : aria(el, 'aria-checked') === 'true' }));

    document.querySelectorAll('input[type="radio"], [role="radio"]').forEach(el =>
        push('radio', el, {
            checked: el instanceof HTMLInputElement ? el.checked : aria(el, 'aria-checked') === 'true',
            group: el.name || null,
        }));

    document.querySelectorAll('input[type="range"], [role="slider"]').forEach(el =>
        push('slider', el, {
            value: (el.value != null && el.value !== '') ? String(el.value) : aria(el, 'aria-valuenow'),
            min: el.min || aria(el, 'aria-valuemin'),
            max: el.max || aria(el, 'aria-valuemax'),
        }));

    document.querySelectorAll('progress, [role="progressbar"]').forEach(el =>
        push('progressbar', el, {
            value: aria(el, 'aria-valuenow') || (el.value != null ? String(el.value) : null),
            min: aria(el, 'aria-valuemin') || '0',
            max: aria(el, 'aria-valuemax') || (el.max != null && el.max !== '' ? String(el.max) : '1'),
        }));

    document.querySelectorAll('meter, [role="meter"]').forEach(el =>
        push('meter', el, {
            value: aria(el, 'aria-valuenow') || (el.value != null ? String(el.value) : null),
            min: aria(el, 'aria-valuemin') || (el.min != null && el.min !== '' ? String(el.min) : '0'),
            max: aria(el, 'aria-valuemax') || (el.max != null && el.max !== '' ? String(el.max) : '1'),
        }));

    // <select> nativo vira combobox com as options anexadas (ação "select"
    // pelo label). Options nativas não são emitidas individualmente.
    document.querySelectorAll('select').forEach(el => {
        const options = [...el.options].slice(0, MAX_OPTIONS).map(o => ({
            label: cleanText(o.label || o.value), value: o.value,
        }));
        push('combobox', el, { value: el.value || null, options }, () => {
            const sel = el.selectedOptions && el.selectedOptions[0];
            return sel ? cleanText(sel.label || sel.value) : null;
        });
    });

    document.querySelectorAll('[role="combobox"], input[list]').forEach(el =>
        push('combobox', el, {
            value: (el.value || '').slice(0, MAX_VALUE) || null,
            expanded: aria(el, 'aria-expanded'),
        }));

    document.querySelectorAll('[role="listbox"]').forEach(el => {
        const opts = [...el.querySelectorAll('[role="option"]')].slice(0, MAX_OPTIONS).map(o => cleanText(o.innerText));
        const sel = el.querySelector('[role="option"][aria-selected="true"]');
        push('listbox', el, {
            options: opts.length ? opts : null,
            selected: sel ? cleanText(sel.innerText) : null,
        });
    });

    // Options soltas (popup fora de listbox/aria-controls); as de dentro da
    // listbox já vieram no array "options" — evita listar tudo duas vezes.
    document.querySelectorAll('[role="option"]').forEach(el => {
        if (el.closest('[role="listbox"]')) return;
        push('option', el, { selected: aria(el, 'aria-selected') });
    });

    document.querySelectorAll('[role="menuitem"], [role="menuitemcheckbox"], [role="menuitemradio"]').forEach(el =>
        push('menuitem', el, { selected: aria(el, 'aria-selected') }));

    document.querySelectorAll('[role="treeitem"]').forEach(el =>
        push('treeitem', el, {
            selected: aria(el, 'aria-selected'),
            expanded: aria(el, 'aria-expanded'),
        }));

    document.querySelectorAll('[role="tab"]').forEach(el =>
        push('tab', el, {
            selected: aria(el, 'aria-selected') === 'true',
            panel_id: aria(el, 'aria-controls'),
        }));

    document.querySelectorAll('dialog[open], [role="dialog"]').forEach(el =>
        push('dialog', el, { modal: el.tagName === 'DIALOG' ? true : aria(el, 'aria-modal') }));
    document.querySelectorAll('[role="alertdialog"]').forEach(el =>
        push('alertdialog', el, { modal: aria(el, 'aria-modal') }));

    /* -------------- Fase 2: conteúdo / estrutura --------------
       Roda DEPOIS dos interativos: com max_items no Python, links/botões/
       campos nunca são empurrados pra fora do orçamento por parágrafos
       e células de tabela. */

    document.querySelectorAll('h1,h2,h3,h4,h5,h6,[role="heading"]').forEach(el => {
        if (el.closest('a[href], button')) return;  // já virou o "text" do link/botão pai
        const level = /^H[1-6]$/.test(el.tagName) ? Number(el.tagName[1]) : Number(aria(el, 'aria-level')) || null;
        push('heading', el, level ? { level } : {});
    });

    document.querySelectorAll('img[alt]:not([alt=""]), [role="img"]').forEach(el =>
        push('img', el, { url: el.currentSrc || el.src || null }));

    // Landmarks: nome explícito ou nome default do papel (com contador).
    document.querySelectorAll('nav, [role="navigation"]').forEach(el =>
        push('navigation', el, {}, aria(el, 'aria-label') || defaultName('navigation'), false));
    document.querySelectorAll('header, [role="banner"]').forEach(el =>
        push('banner', el, {}, aria(el, 'aria-label') || defaultName('banner'), false));
    document.querySelectorAll('footer, [role="contentinfo"]').forEach(el =>
        push('contentinfo', el, {}, aria(el, 'aria-label') || defaultName('contentinfo'), false));
    document.querySelectorAll('aside, [role="complementary"]').forEach(el =>
        push('complementary', el, {}, aria(el, 'aria-label') || defaultName('complementary'), false));
    document.querySelectorAll('[role="search"], search').forEach(el =>
        push('search', el, {}, aria(el, 'aria-label') || defaultName('search'), false));

    // Containers só entram se tiverem nome (sem nome é ruído).
    document.querySelectorAll('form, [role="form"]').forEach(el =>
        push('form', el, {}, aria(el, 'aria-label'), false));
    document.querySelectorAll('[role="region"]').forEach(el =>
        push('region', el, {}, aria(el, 'aria-label'), false));
    ['menu', 'menubar', 'tablist', 'tree', 'list'].forEach(role => {
        document.querySelectorAll(`[role="${role}"]`).forEach(el =>
            push(role, el, {}, aria(el, 'aria-label'), false));
    });
    document.querySelectorAll('ul, ol').forEach(el =>
        push('list', el, {}, aria(el, 'aria-label'), false));

    document.querySelectorAll('[role="tabpanel"]').forEach(el => {
        const tab = el.id ? document.querySelector(`[aria-controls="${el.id}"]`) : null;
        push('tabpanel', el, {}, aria(el, 'aria-label') || (tab ? cleanText(tab.innerText) : null), false);
    });

    // <li> "folha" e sem controles: o resto já foi capturado como link/botão
    // (evita o blob "título + meta + link" duplicado dentro do listitem).
    document.querySelectorAll('li, [role="listitem"]').forEach(el => {
        if (el.querySelector('a[href], button, input, select, textarea, li, [role="button"]')) return;
        push('listitem', el, { position: aria(el, 'aria-posinset') });
    });

    document.querySelectorAll('p').forEach(el => push('paragraph', el));
    document.querySelectorAll('[role="note"], blockquote').forEach(el => push('note', el));
    document.querySelectorAll('figcaption, caption, [role="caption"]').forEach(el => push('caption', el));

    document.querySelectorAll('table, [role="table"], [role="grid"]').forEach(el => {
        const role = aria(el, 'role') === 'grid' ? 'grid' : 'table';
        const cap = el.querySelector('caption');
        push(role, el, {}, aria(el, 'aria-label') || (cap ? cleanText(cap.innerText) : null), false);
    });
    document.querySelectorAll('tr, [role="row"]').forEach(el =>
        push('row', el, {
            position: aria(el, 'aria-rowindex') ||
                      (typeof el.sectionRowIndex === 'number' && el.sectionRowIndex >= 0 ? el.sectionRowIndex + 1 : null),
        }));
    document.querySelectorAll('td, th, [role="gridcell"], [role="columnheader"], [role="rowheader"]').forEach(el => {
        const r = aria(el, 'role');
        const role = r || (el.tagName === 'TH' ? 'columnheader' : 'gridcell');
        push(role, el, { row: aria(el, 'aria-rowindex'), column: aria(el, 'aria-colindex') });
    });

    document.querySelectorAll('[role="alert"]').forEach(el =>
        push('alert', el, { urgency: aria(el, 'aria-live') || 'assertive' }));
    document.querySelectorAll('[role="status"], output').forEach(el =>
        push('status', el, { urgency: aria(el, 'aria-live') || 'polite' }));

    document.querySelectorAll('figure, [role="figure"]').forEach(el =>
        push('figure', el, {}, aria(el, 'aria-label'), false));

    return results;
}
"""

# ---------------------------------------------------------------------------
# Cache da última extração por página. Usado para, a partir de um "id"
# numérico devolvido ao agente, recuperar o role/texto/url originais do item
# — que são então usados para montar um locator do Playwright por role+nome
# acessível (ver _locate_item), em vez de reabrir o elemento por um atributo
# gravado no DOM.
# ---------------------------------------------------------------------------

_LAST_EXTRACTION: dict[int, dict] = {}  # id(page) -> {"kind": ..., "items": [...]}


def _cache_extraction(page: Page, kind: str, items: list[dict]) -> None:
    _LAST_EXTRACTION[id(page)] = {"kind": kind, "items": items}


def extract_page_elements(page: Page, max_items: int = 400) -> list[dict]:
    """
    Extração DOM-based cobrindo todos os roles interativos e de conteúdo do
    ROLE_SCHEMA. Estratégias principais:
      - Fase 1 (interativos) roda antes da Fase 2 (conteúdo): com o teto de
        max_items, links/botões/campos nunca são varridos da lista por
        parágrafos e células.
      - Título curto por elemento (heading interno > aria-label > primeiro
        trecho separado por •/|), resto vai pra "meta".
      - Elementos desabilitados aparecem com "disabled": true (o agente vê
        que existem; execute_item_action bloqueia agir sobre eles).
      - <select> nativo vira combobox com array "options".
      - Toda property declarada no ROLE_SCHEMA que vier do JS é repassada
        (value, checked, options, min/max, expanded, selected, level, url...).
    """
    raw_results = page.evaluate(GENERIC_EXTRACTION_JS, ITEM_ID_ATTR)

    results = []
    for item in raw_results[:max_items]:
        role = item.get("role")
        schema = ROLE_SCHEMA.get(role)
        if not schema or role in SKIP_ROLES:
            continue

        text = clean_text(item.get("text"))
        if not text:
            continue

        out: dict = {"id": item["id"], "role": role, "text": text, "actions": schema["actions"]}

        # Repasse genérico: qualquer property do schema presente no item.
        # "text"/"label" já viraram o campo "text".
        for prop in schema["properties"]:
            if prop in ("text", "label"):
                continue
            value = item.get(prop)
            if value is not None and value != "":
                out[prop] = value

        meta = clean_text(item.get("meta"))
        if meta:
            out["meta"] = meta

        results.append(out)

    _cache_extraction(page, "generic", results)
    return results


# ---------------------------------------------------------------------------
# Extração de resultados orgânicos do Google (título + url + snippet)
# ---------------------------------------------------------------------------

def extract_organic_results(page: Page) -> list[dict]:
    """
    Extrai resultados orgânicos do Google agrupando título + url + snippet
    pelo container DOM real, em vez de uma lista plana de roles. Cada
    resultado ganha um "id" numérico único e o link correspondente é marcado
    no DOM com data-ava-id, para que execute_item_action() consiga clicar
    nele depois (além de "navigate" via url, que não precisa do DOM).
    """
    raw_results = page.evaluate(
        """
        (idAttr) => {
            document.querySelectorAll(`[${idAttr}]`).forEach(el => el.removeAttribute(idAttr));
            let nextId = 1;

            const results = [];
            const seenUrls = new Set();

            const headings = document.querySelectorAll('h3');

            headings.forEach(h3 => {
                const title = h3.innerText.trim();
                if (!title) return;

                const link = h3.closest('a');
                if (!link || !link.href) return;

                const url = link.href;

                if (url.includes('google.com/search') || url.startsWith('javascript:')) return;
                if (seenUrls.has(url)) return;
                seenUrls.add(url);

                let container = link.closest('div[data-hveid], div[jscontroller]') || link.parentElement?.parentElement?.parentElement;
                let snippet = '';

                if (container) {
                    const textNodes = container.querySelectorAll('span, div');
                    for (const node of textNodes) {
                        const text = node.innerText?.trim();
                        if (text && text.length > 40 && !text.includes(title) && node.children.length === 0) {
                            snippet = text;
                            break;
                        }
                    }
                }

                const id = nextId++;
                link.setAttribute(idAttr, String(id));

                results.push({ id, title, url, snippet });
            });

            return results;
        }
        """,
        ITEM_ID_ATTR,
    )

    # Cada resultado orgânico é sempre "somente leitura + navegável" para o agente
    for item in raw_results:
        item["role"] = "result"
        item["actions"] = ["read", "navigate", "click"]

    _cache_extraction(page, "organic", raw_results)
    return raw_results


def _goto_and_wait(page: Page, url: str, wait_selector: str = "body", timeout: int = 10000):
    """
    Navega e espera de forma otimizada, com logging de progresso para
    diagnosticar travamentos.
    """
    print(f"[goto] Navegando para: {url}")
    t0 = time.time()

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        print(f"[goto] domcontentloaded em {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"[goto] AVISO: timeout/erro ao navegar ({e}). Seguindo com o que carregou.")

    try:
        page.wait_for_selector(wait_selector, timeout=timeout)
        print(f"[wait] Seletor '{wait_selector}' encontrado em {time.time() - t0:.2f}s (total)")
    except Exception as e:
        print(f"[wait] AVISO: seletor '{wait_selector}' não apareceu a tempo ({e}). Seguindo.")


def fetch_and_extract(query_or_url: str, page: Page, is_url: bool = False) -> list[dict]:
    if is_url:
        url = query_or_url
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        _goto_and_wait(page, url, wait_selector="body")
        print("[extract] Extraindo elementos da página...")
        result = extract_page_elements(page)
        print(f"[extract] {len(result)} elementos extraídos.")
        return result
    else:
        url = BASE_URL + quote(query_or_url)
        _goto_and_wait(page, url, wait_selector="h3")
        print("[extract] Extraindo resultados orgânicos...")
        result = extract_organic_results(page)
        print(f"[extract] {len(result)} resultados extraídos.")
        return result



def search_google(query: str, page: Page) -> list[dict]:
    return fetch_and_extract(query, page, is_url=False)


def search_link(url: str, page: Page) -> list[dict]:
    return fetch_and_extract(url, page, is_url=True)


# ---------------------------------------------------------------------------
# Execução de ações sobre um item retornado pela extração (via seu "id")
# ---------------------------------------------------------------------------


# Roles do ROLE_SCHEMA que não são roles ARIA "de verdade" e por isso não
# existem para o page.get_by_role() do Playwright — precisam de um mapeamento
# manual para o role real que o elemento tem no DOM.
_ROLE_TO_ARIA = {
    "result": "link",  # resultado orgânico do Google = <a> contendo um <h3>
}


def _accessible_name(item: dict) -> str:
    # extract_organic_results usa "title" em vez de "text" para o rótulo.
    return item.get("text") or item.get("title") or ""


def _locate_item(page: Page, item: dict, timeout: int) -> Locator:
    """
    Monta um locator do Playwright por role ARIA + nome acessível, a partir
    das propriedades que a extração já capturou — em vez de reabrir o
    elemento por um id gravado no DOM.

    Por que isso é mais robusto: page.get_by_role() não aponta pra um nó
    específico, ele é uma QUERY que o Playwright reexecuta a cada chamada
    (.click(), .evaluate() etc.), com auto-retry embutido até o timeout.
    Então, se a página re-renderizar entre a extração e a ação — coisa que
    SPAs como a huggingface.co fazem continuamente, não só uma vez no load
    — o locator naturalmente encontra o elemento novo (mesmo role, mesmo
    nome), sem precisar que nenhum atributo tenha sobrevivido à troca de nó.
    """
    role = _ROLE_TO_ARIA.get(item.get("role"), item.get("role"))
    name = _accessible_name(item)
    url = item.get("url")

    loc = page.get_by_role(role, name=name, exact=True)
    try:
        count = loc.count()
    except Exception:
        count = 0

    if count == 0:
        # o texto extraído pode ter sido truncado/normalizado (MAX_TEXT=160,
        # espaços colapsados etc.) e não bater 100% com o nome acessível
        # real — tenta de novo como substring.
        loc = page.get_by_role(role, name=name, exact=False)
        try:
            count = loc.count()
        except Exception:
            count = 0

    if count > 1 and url:
        # mais de um elemento com o mesmo role+nome (ex.: "Models" existe
        # como link no menu desktop E como botão no menu mobile — ali o role
        # já desempata; isto aqui cobre o caso de dois <a> com texto igual).
        # Não dá pra desempatar via seletor CSS [href="..."] porque o
        # atributo no DOM pode estar relativo ("/models") enquanto a url
        # extraída é absoluta — por isso compara o href RESOLVIDO de cada
        # candidato, um por um.
        for i in range(count):
            cand = loc.nth(i)
            try:
                href = cand.evaluate("el => el.href || null", timeout=timeout)
            except Exception:
                continue
            if href == url:
                return cand
        # nenhum bateu a url exatamente: segue com o primeiro (melhor palpite)

    return loc.first if count > 0 else loc


def execute_item_action(
    page: Page,
    item_id: int,
    action: str,
    value: Optional[str] = None,
    timeout: int = 5000,
) -> dict:
    """
    Executa uma ação num item retornado por extract_page_elements() /
    extract_organic_results(), re-localizando-o por role ARIA + nome
    acessível (ver _locate_item) — não por um id gravado no DOM, que não
    sobrevive a re-renders da SPA.

    Ações suportadas (cada role anuncia as dele em "actions"):
      click           — link, button, tab, menuitem, option, row, gridcell...
      read            — qualquer elemento (funciona até em desabilitado)
      type / clear    — textbox, searchbox, combobox, textarea
      toggle          — checkbox, switch (reporta "checked" depois do clique)
      select          — combobox <select>/ARIA, listbox, option, radio
                        (value = label da opção)
      set_value       — slider nativo ou ARIA (value = número alvo)
      expand/collapse — treeitem, menu, combobox (respeita aria-expanded atual)
      close           — dialog/alertdialog (Escape / <dialog>.close())
      submit          — searchbox (Enter) ou form (requestSubmit)
      press           — qualquer tecla/combo (value = "Enter", "Escape",
                        "Tab", "ArrowDown", "Control+A"...) sobre o elemento
      navigate        — link/resultado orgânico (via href)

    Se o id não estiver na última extração desta página (extract_page_elements
    ou extract_organic_results ainda não rodou, ou rodou em outra page), o
    erro pede pra rodar a extração de novo — não tem role/texto pra procurar.
    """
    cached = _LAST_EXTRACTION.get(id(page))
    item = next((i for i in cached["items"] if i.get("id") == item_id), None) if cached else None

    if item is None:
        return {
            "success": False,
            "error": f"id={item_id} não está na última extração desta página "
                     f"(rode extract_page_elements/search_link ou "
                     f"extract_organic_results/search_google de novo antes de agir).",
        }

    locator = _locate_item(page, item, timeout)

    try:
        count = locator.count()
    except Exception as e:
        return {"success": False, "error": f"falha ao localizar elemento: {e}"}

    if count == 0:
        return {
            "success": False,
            "error": f"nenhum elemento role={item.get('role')!r} com nome acessível "
                     f"{_accessible_name(item)!r} foi encontrado na página atual "
                     f"(pode ter sido removido/renomeado de verdade — rode a extração de novo).",
        }

    try:
        tag = (locator.evaluate("el => el.tagName", timeout=timeout) or "").lower()
        input_type = (locator.evaluate("el => el.type || ''", timeout=timeout) or "") if tag == "input" else ""
        role_attr = locator.evaluate("el => el.getAttribute('role')", timeout=timeout)

        # "read" funciona mesmo em elemento desabilitado; ações interativas, não.
        if action != "read":
            is_disabled = locator.evaluate(
                "el => el.disabled === true || el.getAttribute('aria-disabled') === 'true'",
                timeout=timeout,
            )
            if is_disabled:
                return {"success": False, "error": f"elemento id={item_id} está desabilitado"}

        if action == "click":
            locator.click(timeout=timeout)
            result = {"clicked": True}

        elif action == "read":
            if tag in ("input", "textarea", "select"):
                text = locator.input_value(timeout=timeout)
            else:
                text = locator.inner_text(timeout=timeout)
            result = {"text": clean_text(text)}

        elif action == "type":
            if value is None:
                return {"success": False, "error": "ação 'type' exige o parâmetro 'value'"}
            locator.fill(value, timeout=timeout)
            result = {"typed": value}

        elif action == "clear":
            locator.fill("", timeout=timeout)
            result = {"cleared": True}

        elif action == "toggle":
            locator.click(timeout=timeout)
            try:
                checked = locator.is_checked()
            except Exception:
                checked = locator.get_attribute("aria-checked") == "true"
            result = {"checked": checked}

        elif action == "select":
            if value is None:
                return {"success": False, "error": "ação 'select' exige o parâmetro 'value'"}

            if tag == "select":
                # <select> nativo: casa por label/value numa única passada,
                # sem depender de timeout por tentativa.
                idx = locator.evaluate(
                    """(el, wanted) => {
                        const norm = (s) => (s || '').toString().trim().toLowerCase();
                        return Array.from(el.options).findIndex(
                            (o) => norm(o.label) === norm(wanted) || norm(o.value) === norm(wanted));
                    }""",
                    str(value),
                )
                if idx is None or idx < 0:
                    return {"success": False,
                            "error": f"opção '{value}' não existe no <select> id={item_id}"}
                locator.select_option(index=idx, timeout=timeout)
                selected = locator.evaluate(
                    "el => el.selectedOptions[0] ? (el.selectedOptions[0].label || el.selectedOptions[0].value) : null"
                )
                result = {"selected": selected}

            elif input_type == "radio" or role_attr == "radio":
                locator.click(timeout=timeout)
                result = {"checked": True, "selected": value}

            elif role_attr == "option":
                locator.click(timeout=timeout)
                result = {"selected": value}

            else:
                # combobox/listbox ARIA: procura a option no popup controlado
                # (aria-controls), dentro do próprio elemento ou solta na página.
                option_loc: Optional[Locator] = None
                candidates = [locator.locator('[role="option"]', has_text=value)]
                controls_id = locator.get_attribute("aria-controls")
                if controls_id:
                    candidates.append(page.locator(f'[id="{controls_id}"] [role="option"]', has_text=value))
                candidates.append(page.locator('[role="option"]', has_text=value))
                for cand in candidates:
                    try:
                        if cand.count() > 0 and cand.first.is_visible():
                            option_loc = cand.first
                            break
                    except Exception:
                        continue
                if option_loc is None:
                    return {"success": False,
                            "error": f"nenhuma option com texto '{value}' visível (abra a lista com 'expand' antes?)"}
                option_loc.click(timeout=timeout)
                result = {"selected": value}

        elif action == "set_value":
            if value is None:
                return {"success": False, "error": "ação 'set_value' exige o parâmetro 'value'"}

            if tag == "input" and input_type == "range":
                # Slider nativo: seta o value e dispara input/change (o setter
                # via evaluate contorna controlled inputs do React etc.).
                locator.evaluate(
                    """(el, v) => {
                        el.value = v;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }""",
                    str(value),
                )
                result = {"value": locator.input_value(timeout=timeout)}
            else:
                # Slider ARIA: foca e caminha com setas até o alvo (passo 1,
                # máx. 100 pressionadas — quem atualiza é o listener do site).
                try:
                    target = float(value)
                except ValueError:
                    return {"success": False, "error": f"set_value espera número, recebi '{value}'"}
                now_raw = locator.get_attribute("aria-valuenow")
                now = float(now_raw) if now_raw else 0.0
                presses = min(int(abs(target - now)), 100)
                key = "ArrowRight" if target > now else "ArrowLeft"
                locator.focus(timeout=timeout)
                for _ in range(presses):
                    locator.press(key, timeout=timeout)
                result = {"value": locator.get_attribute("aria-valuenow")}

        elif action in ("expand", "collapse"):
            want_open = action == "expand"

            def read_expanded() -> Optional[bool]:
                return locator.evaluate(
                    """(el) => {
                        const target = el.hasAttribute('aria-expanded')
                            ? el
                            : el.querySelector('[aria-expanded]');
                        return target ? target.getAttribute('aria-expanded') === 'true' : null;
                    }"""
                )

            current = read_expanded()
            if current is not None and current is want_open:
                result = {"expanded": current, "changed": False}  # já está no estado pedido
            else:
                # Prefere o próprio elemento (tem aria-expanded); senão o
                # "expander" filho (chevron); senão clica no elemento mesmo.
                if locator.evaluate("el => el.hasAttribute('aria-expanded')"):
                    locator.click(timeout=timeout)
                elif locator.locator('[aria-expanded]').count() > 0:
                    locator.locator('[aria-expanded]').first.click(timeout=timeout)
                else:
                    locator.click(timeout=timeout)
                result = {"expanded": read_expanded(), "changed": True}

        elif action == "close":
            if tag == "dialog":
                locator.evaluate("el => el.close()")
                result = {"closed": True}
            else:
                # Dialog ARIA: Escape primeiro; se persistir, tenta botão de fechar.
                locator.press("Escape", timeout=timeout)
                if locator.count() > 0 and locator.is_visible():
                    close_btn = locator.locator(
                        'button:has-text("close"), button:has-text("fechar"), '
                        '[aria-label*="close" i], [aria-label*="fechar" i]'
                    ).first
                    if close_btn.count() > 0:
                        close_btn.click(timeout=timeout)
                result = {"closed": locator.count() == 0 or not locator.is_visible()}

        elif action == "submit":
            if tag == "form":
                locator.evaluate("el => el.requestSubmit ? el.requestSubmit() : el.submit()")
            else:
                locator.press("Enter", timeout=timeout)
            result = {"submitted": True}

        elif action == "press":
            # Tecla (ou combo) do teclado sobre o elemento focado, via
            # Locator.press — foca o elemento e dispara o keydown/keyup real,
            # então funciona pra Enter num campo de busca, Escape num popup,
            # Tab pra mover o foco, setas em listas/menus, atalhos tipo
            # "Control+A"/"Shift+Tab" etc.
            if value is None:
                return {"success": False,
                        "error": "ação 'press' exige o parâmetro 'value' com a tecla "
                                 "(ex.: 'Enter', 'Escape', 'Tab', 'ArrowDown', 'Control+A')"}
            locator.press(value, timeout=timeout)
            result = {"pressed": value}

        elif action == "navigate":
            url = locator.evaluate("el => el.href || null")
            if not url:
                return {"success": False, "error": "elemento não tem href para navegar"}
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            result = {"navigated_to": url}

        else:
            return {"success": False, "error": f"ação '{action}' não suportada"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    return {"success": True, **result}


def press_key(page: Page, key: str) -> dict:
    """
    Aperta uma tecla (ou combo, ex. "Control+A") no nível da página inteira,
    via page.keyboard — sem precisar de um item extraído nem de saber onde
    ele está. Útil pra:
      - atalhos globais (Escape pra fechar um popup que a extração nem
        capturou, Tab pra mover o foco antes de decidir o próximo passo);
      - apertar Enter logo depois de um action="type" num campo, já que
        Locator.fill() (usado pelo "type") seta o valor direto e não dispara
        o keydown/keyup de um Enter de verdade.
    Quem quiser apertar uma tecla NUM elemento específico (ex.: Enter só
    depois de garantir que o foco está no campo certo) deve usar
    execute_item_action(page, item_id, "press", value=key) em vez desta.
    """
    try:
        page.keyboard.press(key)
        return {"success": True, "pressed": key}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Gerenciamento do Chrome real do usuário — via Shell Command API
# ---------------------------------------------------------------------------
#
# O browser_agent.py não abre mais o processo do Chrome diretamente (nada de
# subprocess.Popen aqui). Quem faz isso agora é o shell_command_api.py, que
# roda como serviço HTTP na máquina onde o Chrome real do usuário vive.
# O Playwright continua conectando via CDP normalmente (connect_over_cdp) —
# só a etapa "abrir/derrubar o processo" saiu daqui.

SHELL_API_URL = os.getenv("AVA_SHELL_API_URL", "http://localhost:4005")


def _shell_command(command: str, params: Optional[dict] = None, timeout: float = 40) -> dict:
    """
    Chama POST /command no Shell Command API e devolve o campo 'result'.

    Lança ConnectionError se o serviço não estiver acessível (ex.: não está
    rodando na máquina do usuário) e RuntimeError se o comando em si falhou
    do lado do serviço.
    """
    try:
        resp = requests.post(
            f"{SHELL_API_URL}/command",
            json={"command": command, "params": params or {}},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Não foi possível alcançar o Shell Command API em {SHELL_API_URL} "
            f"(ele está rodando na máquina do usuário?): {e}"
        ) from e

    if resp.status_code != 200:
        raise RuntimeError(f"Shell Command API retornou {resp.status_code}: {resp.text}")

    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"Comando '{command}' falhou no Shell Command API: {payload.get('error')}")

    return payload.get("result") or {}


def is_port_open(port: int) -> bool:
    """Verifica (via Shell Command API) se uma porta local está aberta na máquina do usuário."""
    result = _shell_command("is_port_open", {"port": port})
    return bool(result.get("open"))


def launch_chrome_with_debugging(port: int = 9222, profile_directory: str = "Default") -> dict:
    """
    Pede ao Shell Command API para abrir (ou reaproveitar) o Chrome real do
    usuário com debugging remoto. A cópia do profile e o subprocess.Popen
    acontecem do lado do serviço, não aqui.
    """
    result = _shell_command(
        "launch_chrome_debug",
        {"port": port, "profile_directory": profile_directory},
    )
    if result.get("already_running"):
        print(f"[agent] Chrome já está rodando na porta {port} (reaproveitado via Shell Command API).")
    else:
        print(f"[agent] Chrome aberto via Shell Command API na porta {port} (pid remoto {result.get('pid')}).")
    return result


def kill_chrome_with_debugging(port: int = 9222) -> dict:
    """Pede ao Shell Command API para derrubar o Chrome que ele abriu nesta porta."""
    return _shell_command("kill_chrome_debug", {"port": port})


# ===========================================================================
# BIBLIOTECA — sessão de navegador persistente
# ===========================================================================
#
# Tudo acima é a "engine" (extração de elementos, execução de ações,
# gerenciamento do processo do Chrome). Daqui pra baixo é a camada que
# transforma isso numa biblioteca fácil de usar:
#
#   - WebAgent: uma sessão que abre o Chrome + conecta o Playwright UMA VEZ
#     e mantém a mesma `page` viva entre comandos. Nada fecha sozinho — só
#     o método/comando `close()` / `fechar()` derruba o browser.
#   - Funções soltas em português (abrir, buscar, estrutura, clicar, digitar,
#     etc.) que usam uma instância padrão de WebAgent por trás, criada na
#     primeira chamada — pra poder simplesmente importar e usar sem
#     instanciar nada manualmente.
#
# Exemplo de uso (num script, notebook ou REPL — a sessão sobrevive entre
# chamadas, só o comando fechar() derruba o Chrome):
#
#   import browser_agent as ba
#
#   itens = ba.abrir("huggingface.co")           # navega + extrai
#   alvo = ba.encontrar(itens, role="link", texto=r"models?")
#   ba.clicar(alvo["id"])
#
#   itens = ba.estrutura()                       # re-lê a página ATUAL,
#                                                 # sem navegar de novo
#   campo = ba.encontrar(itens, role=("textbox", "searchbox"),
#                         texto=r"filter by name")
#   ba.digitar(campo["id"], "LFM2.5 8B A1B")
#   ba.apertar(campo["id"], "Enter")
#
#   ba.estrutura()                                # lê de novo depois do Enter
#   ...
#   ba.fechar()                                   # só aqui o Chrome fecha
#
# ===========================================================================


class WebAgent:
    """
    Uma sessão de navegador persistente: abre/reaproveita o Chrome com
    debugging remoto, conecta o Playwright (com stealth) e mantém uma única
    `page` viva. Todos os métodos operam sobre essa página até que `close()`
    seja chamado — nenhuma ação fecha o browser sozinha.
    """

    def __init__(
        self,
        port: int = 9222,
        profile_directory: str = "Default",
        block_resources: bool = True,
    ):
        self.port = port
        self.profile_directory = profile_directory
        self.page: Optional[Page] = None

        # True só quando FOI ESTE agente quem pediu ao Shell Command API pra
        # abrir o Chrome (isto é, ele não estava rodando antes). Usado em
        # close() pra decidir se derrubamos o processo remoto ou se deixamos
        # como estava (Chrome que já estava aberto por outra sessão/usuário).
        self._launched_remotely = False
        self._stealth_cm = None
        self._playwright = None
        self._browser = None
        self._context = None

        self._start(block_resources)

    # -- ciclo de vida ------------------------------------------------

    def _start(self, block_resources: bool) -> None:
        print("[agent] Pedindo ao Shell Command API para abrir o Chrome com debugging...")
        launch_result = launch_chrome_with_debugging(self.port, self.profile_directory)
        self._launched_remotely = not launch_result.get("already_running", False)
        print("[agent] Chrome pronto. Conectando via CDP...")

        # Stealth().use_sync(...) é um context manager; entramos nele
        # manualmente (sem "with") pra manter o playwright vivo entre
        # chamadas, e só saímos dele de verdade em close().
        self._stealth_cm = Stealth().use_sync(sync_playwright())
        self._playwright = self._stealth_cm.__enter__()

        self._browser = self._playwright.chromium.connect_over_cdp(f"http://localhost:{self.port}")
        self._context = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context()
        self._context.set_default_timeout(10000)
        self._context.set_default_navigation_timeout(10000)

        self.page = self._context.new_page()
        if block_resources:
            self.page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,css}", lambda route: route.abort())
        print("[agent] Sessão pronta.")

    def close(self) -> None:
        """Fecha o Playwright e, se foi este agente quem abriu o Chrome, pede ao
        Shell Command API para derrubar o processo remoto. Único comando que
        encerra a sessão."""
        try:
            if self._browser:
                self._browser.close()
        except Exception as e:
            print(f"[agent] aviso ao fechar browser: {e}")

        try:
            if self._stealth_cm:
                self._stealth_cm.__exit__(None, None, None)
        except Exception as e:
            print(f"[agent] aviso ao encerrar playwright: {e}")

        if self._launched_remotely:
            try:
                kill_chrome_with_debugging(self.port)
            except Exception as e:
                print(f"[agent] aviso ao derrubar Chrome via Shell Command API: {e}")

        self.page = None
        print("[agent] Sessão encerrada.")

    def new_page(self) -> Page:
        """Abre uma nova aba na mesma sessão e passa a usá-la como página atual."""
        self.page = self._context.new_page()
        return self.page

    # -- navegação + extração ------------------------------------------

    def open(self, url: str) -> list[dict]:
        """Navega para uma URL e devolve os elementos extraídos da página resultante."""
        return fetch_and_extract(url, self.page, is_url=True)

    def search(self, query: str) -> list[dict]:
        """Faz uma busca no Google e devolve os resultados orgânicos (título/url/snippet)."""
        return fetch_and_extract(query, self.page, is_url=False)

    def structure(self, max_items: int = 400) -> list[dict]:
        """
        Re-extrai a estrutura da página ATUAL (sem navegar) — comando central
        pra ver "como a página está agora" depois de um clique, digitação,
        scroll, espera, etc.
        """
        return extract_page_elements(self.page, max_items=max_items)

    def results(self) -> list[dict]:
        """Re-extrai resultados orgânicos do Google na página atual (sem navegar)."""
        return extract_organic_results(self.page)

    # -- ações sobre itens extraídos ------------------------------------

    def act(self, item_id: int, action: str, value: Optional[str] = None, timeout: int = 5000) -> dict:
        """Ação genérica — ver ROLE_SCHEMA / execute_item_action para a lista completa."""
        return execute_item_action(self.page, item_id, action, value, timeout)

    def click(self, item_id: int) -> dict:
        return self.act(item_id, "click")

    def type(self, item_id: int, text: str) -> dict:
        return self.act(item_id, "type", text)

    def clear(self, item_id: int) -> dict:
        return self.act(item_id, "clear")

    def press(self, item_id: int, key: str) -> dict:
        return self.act(item_id, "press", key)

    def toggle(self, item_id: int) -> dict:
        return self.act(item_id, "toggle")

    def select(self, item_id: int, value: str) -> dict:
        return self.act(item_id, "select", value)

    def set_value(self, item_id: int, value: str) -> dict:
        return self.act(item_id, "set_value", value)

    def expand(self, item_id: int) -> dict:
        return self.act(item_id, "expand")

    def collapse(self, item_id: int) -> dict:
        return self.act(item_id, "collapse")

    def submit(self, item_id: int) -> dict:
        return self.act(item_id, "submit")

    def navigate(self, item_id: int) -> dict:
        return self.act(item_id, "navigate")

    def close_dialog(self, item_id: int) -> dict:
        return self.act(item_id, "close")

    def press_global(self, key: str) -> dict:
        """Aperta uma tecla no nível da página (sem precisar de um item extraído)."""
        return press_key(self.page, key)

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    # -- utilidade --------------------------------------------------

    @staticmethod
    def find(items: list[dict], role: Optional[str | tuple | list] = None,
              text: Optional[str] = None) -> Optional[dict]:
        """Primeiro item que casa com role (str ou tupla/lista de roles) e/ou regex no texto."""
        roles = (role,) if isinstance(role, str) else role
        for it in items:
            if roles and it.get("role") not in roles:
                continue
            if text and not re.search(text, it.get("text", ""), re.IGNORECASE):
                continue
            return it
        return None


# ---------------------------------------------------------------------------
# Sessão padrão + funções soltas em português — para uso direto sem
# instanciar WebAgent manualmente. Criada de forma preguiçosa na primeira
# chamada; fechada explicitamente só por fechar().
# ---------------------------------------------------------------------------

_default_agent: Optional["WebAgent"] = None


def _get_default() -> "WebAgent":
    global _default_agent
    if _default_agent is None:
        _default_agent = WebAgent()
    return _default_agent


def abrir(url: str) -> list[dict]:
    """Navega para uma URL e devolve os elementos extraídos."""
    return _get_default().open(url)


def buscar(query: str) -> list[dict]:
    """Busca no Google e devolve os resultados orgânicos."""
    return _get_default().search(query)


def estrutura(max_items: int = 400) -> list[dict]:
    """Re-extrai a estrutura da página atual (sem navegar)."""
    return _get_default().structure(max_items)


def resultados() -> list[dict]:
    """Re-extrai resultados orgânicos do Google na página atual."""
    return _get_default().results()


def agir(item_id: int, acao: str, valor: Optional[str] = None) -> dict:
    return _get_default().act(item_id, acao, valor)


def clicar(item_id: int) -> dict:
    return _get_default().click(item_id)


def digitar(item_id: int, texto: str) -> dict:
    return _get_default().type(item_id, texto)


def limpar(item_id: int) -> dict:
    return _get_default().clear(item_id)


def apertar(item_id: int, tecla: str) -> dict:
    return _get_default().press(item_id, tecla)


def apertar_global(tecla: str) -> dict:
    return _get_default().press_global(tecla)


def alternar(item_id: int) -> dict:
    return _get_default().toggle(item_id)


def selecionar(item_id: int, valor: str) -> dict:
    return _get_default().select(item_id, valor)


def definir_valor(item_id: int, valor: str) -> dict:
    return _get_default().set_value(item_id, valor)


def expandir(item_id: int) -> dict:
    return _get_default().expand(item_id)


def colapsar(item_id: int) -> dict:
    return _get_default().collapse(item_id)


def enviar(item_id: int) -> dict:
    return _get_default().submit(item_id)


def navegar(item_id: int) -> dict:
    return _get_default().navigate(item_id)


def fechar_dialogo(item_id: int) -> dict:
    return _get_default().close_dialog(item_id)


def esperar(segundos: float) -> None:
    _get_default().wait(segundos)


def encontrar(items: list[dict], role: Optional[str | tuple | list] = None,
              texto: Optional[str] = None) -> Optional[dict]:
    return WebAgent.find(items, role=role, text=texto)


def nova_aba() -> Page:
    return _get_default().new_page()


def fechar() -> None:
    """Único comando que realmente fecha o browser (Playwright + processo do Chrome)."""
    global _default_agent
    if _default_agent is not None:
        _default_agent.close()
        _default_agent = None
    else:
        print("[agent] nenhuma sessão ativa para fechar.")
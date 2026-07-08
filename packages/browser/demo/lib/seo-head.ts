// seo-head.ts — client-only imperative <head> sync for SPA navigation.
//
// The prerendered chapter / landing / hub pages (scripts/prerender.entry.tsx)
// already carry the correct <head> on DIRECT entry — which is what crawlers and
// social-card unfurlers fetch. But after CLIENT-SIDE navigation within the SPA,
// the served document's <head> (canonical, description, OG/Twitter, JSON-LD)
// would otherwise stay frozen on the first-loaded route. This keeps the full
// head consistent with the active route for JS-executing crawlers and any
// in-session/social state.
//
// Browser-only: applySeoHead no-ops under SSR (typeof document === 'undefined'),
// so it is safe to import from route components. It is intentionally NOT imported
// by the prerender entry (which emits the head as a string instead).

import { htmlLang } from './i18n';
import { ogLocale, type HreflangAlternate, type PageSeo } from './seo-metadata';

// Shared id for the structured-data block so the client updates the SAME node
// the prerender emitted (see jsonLdScript in scripts/prerender.entry.tsx) rather
// than appending a second, conflicting JSON-LD script.
const JSONLD_ID = 'seo-jsonld';

function upsertMeta(attr: 'name' | 'property', key: string, content: string): void {
  let el = document.head.querySelector<HTMLMetaElement>(`meta[${attr}="${key}"]`);
  if (!el) {
    el = document.createElement('meta');
    el.setAttribute(attr, key);
    document.head.appendChild(el);
  }
  el.setAttribute('content', content);
}

function upsertLink(rel: string, href: string): void {
  let el = document.head.querySelector<HTMLLinkElement>(`link[rel="${rel}"]`);
  if (!el) {
    el = document.createElement('link');
    el.setAttribute('rel', rel);
    document.head.appendChild(el);
  }
  el.setAttribute('href', href);
}

/**
 * Reconcile the `<link rel="alternate" hreflang=…>` set with the active page's
 * alternates IN PLACE (update-or-create per hreflang, remove stale ones), the
 * same idempotent style as the canonical/meta handling above. Passing [] —
 * e.g. for the English-only /chat — clears any alternates left behind by a
 * previously-viewed bilingual page.
 */
function syncAlternateLinks(alternates: HreflangAlternate[]): void {
  const wanted = new Map(alternates.map((a) => [a.hreflang, a.href]));
  const existing = Array.from(document.head.querySelectorAll<HTMLLinkElement>('link[rel="alternate"][hreflang]'));
  for (const el of existing) {
    const hreflang = el.getAttribute('hreflang') ?? '';
    const href = wanted.get(hreflang);
    if (href === undefined) {
      el.remove();
      continue;
    }
    el.setAttribute('href', href);
    wanted.delete(hreflang);
  }
  for (const [hreflang, href] of wanted) {
    const el = document.createElement('link');
    el.setAttribute('rel', 'alternate');
    el.setAttribute('hreflang', hreflang);
    el.setAttribute('href', href);
    document.head.appendChild(el);
  }
}

function upsertJsonLd(payload: object | object[] | null): void {
  let el = document.getElementById(JSONLD_ID) as HTMLScriptElement | null;
  if (payload == null) {
    el?.remove();
    return;
  }
  if (!el) {
    el = document.createElement('script');
    el.type = 'application/ld+json';
    el.id = JSONLD_ID;
    document.head.appendChild(el);
  }
  el.textContent = JSON.stringify(payload);
}

/**
 * Synchronize the document <head> with the active route's SEO. Updates the
 * <html lang>, title, canonical link, hreflang alternates, description, Open
 * Graph + Twitter tags, and the JSON-LD block IN PLACE — matching the tags the
 * prerender emitted, so no duplicates are created. Pass `jsonLd: null` for
 * routes with no structured data (e.g. /chat).
 */
export function applySeoHead(seo: PageSeo, jsonLd: object | object[] | null): void {
  if (typeof document === 'undefined') return;
  document.documentElement.lang = htmlLang(seo.locale);
  document.title = seo.title;
  upsertLink('canonical', seo.canonical);
  syncAlternateLinks(seo.alternates);
  upsertMeta('property', 'og:locale', ogLocale(seo.locale));
  upsertMeta('name', 'description', seo.description);
  upsertMeta('property', 'og:type', seo.ogType);
  upsertMeta('property', 'og:url', seo.canonical);
  upsertMeta('property', 'og:title', seo.title);
  upsertMeta('property', 'og:description', seo.description);
  upsertMeta('property', 'og:image', seo.image);
  upsertMeta('name', 'twitter:title', seo.title);
  upsertMeta('name', 'twitter:description', seo.description);
  upsertMeta('name', 'twitter:image', seo.image);
  upsertJsonLd(jsonLd);
}

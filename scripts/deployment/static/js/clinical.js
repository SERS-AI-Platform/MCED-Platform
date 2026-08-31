/* SERS Clinical Webapp - Minimal JS */

// Toggle collapsible sections
function toggleCollapsible(btn) {
    btn.classList.toggle('open');
    const content = btn.nextElementSibling;
    content.classList.toggle('open');
    btn.setAttribute('aria-expanded', String(content.classList.contains('open')));
}

// Language toggle
function toggleLang() {
    const current = document.cookie.match(/sers_lang=(\w+)/);
    const newLang = (current && current[1] === 'ko') ? 'en' : 'ko';
    document.cookie = `sers_lang=${newLang};path=/;max-age=${86400 * 365}`;
    location.reload();
}

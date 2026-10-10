from pathlib import Path
from playwright.sync_api import sync_playwright
BASE=Path(__file__).resolve().parent
css=(BASE/'strategy.css').read_text()
core=(BASE/'strategy-core.js').read_text()
fixtures=(BASE/'fixtures.js').read_text()
ui=(BASE/'strategy-ui.js').read_text()
issues=[]

def build(case='J'):
    return f'''<!doctype html><html lang="el"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{css}</style></head><body><div class="site"><header class="topline"><span class="topmark"></span><span class="app-name">Οστεοπόρωση</span><span class="tag">ΣΥΝΘΕΤΙΚΗ ΠΡΟΕΠΙΣΚΟΠΗΣΗ</span></header><main id="root"></main><footer class="footer">Δοκιμή μόνο</footer></div><script>window.__STRATEGY_CASE__={case!r};</script><script>{core}</script><script>{fixtures}</script><script>{ui}</script></body></html>'''

def load(page,case='J'):
    page.set_content(build(case),wait_until='load')
    page.wait_for_selector('#root .surface')

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1280,'height':840},device_scale_factor=1)
    page.on('pageerror',lambda error: issues.append(str(error)))
    load(page)
    assert page.get_by_text('Ποια θεραπευτική προσέγγιση αξίζει να εξετάσουμε;').is_visible()
    assert page.get_by_text('Δεν έχει καταγραφεί προηγούμενη θεραπεία').is_visible()
    page.screenshot(path=str(BASE/'preview-J-desktop.png'),full_page=True)
    page.get_by_role('button',name='Να δούμε τι ταιριάζει').click()
    assert page.get_by_text('Έχει προηγηθεί θεραπεία οστεοπόρωσης;').is_visible()
    page.get_by_role('button',name='Επιβεβαιωμένα, όχι').click()
    assert page.get_by_text('Ποιες προσεγγίσεις αξίζει να συζητήσουμε;').is_visible()
    page.get_by_role('button',name='Κράτησέ το στη συζήτηση').click()
    assert page.get_by_text('✓ Κρατήθηκε στη συζήτηση').is_visible()
    page.get_by_role('button',name='Πού καταλήξαμε σήμερα;').click()
    page.locator('textarea[name=recommendation]').fill('Να συζητηθούν οι διαθέσιμες επιλογές.')
    page.locator('select[name=patient_preference]').select_option('undecided')
    page.locator('select[name=decision]').select_option('deferred')
    page.locator('textarea[name=rationale]').fill('Χρειάζεται ενημέρωση.')
    page.get_by_role('button',name='Δες την καταγεγραμμένη κατάληξη').click()
    assert page.get_by_text('Αναβολή',exact=True).is_visible()
    page.get_by_role('button',name='Τι θα μεταφερθεί στο Σχέδιο;').click()
    assert page.get_by_text('"authoritative_write": false').is_visible()
    print('J full flow: PASS')

    load(page,'JD')
    assert page.get_by_text('Τι σημαίνει η προηγούμενη Prolia για τη σημερινή απόφαση;').is_visible()
    page.screenshot(path=str(BASE/'preview-JD-desktop.png'),full_page=True)
    page.get_by_role('button',name='Δες τη θεραπευτική πορεία').click()
    assert page.get_by_text('Τελευταία πραγματική χορήγηση: δεν έχει επιβεβαιωθεί.').is_visible()
    page.get_by_role('button',name='Ακολούθησε άλλη αγωγή').click()
    assert page.get_by_text('Έχει καταγραφεί μεταγενέστερη θεραπεία.').is_visible()
    page.get_by_role('button',name='Τι σημαίνει για την απόφαση;').click()
    assert page.get_by_text('Συνέχεια μετά την Prolia').is_visible()
    page.get_by_role('button',name='Πού καταλήξαμε σήμερα;').click()
    page.locator('select[name=decision]').select_option('pending')
    page.get_by_role('button',name='Δες την καταγεγραμμένη κατάληξη').click()
    assert page.get_by_text('μη επιβεβαιωμένη τελευταία πραγματική χορήγηση').is_visible()
    print('JD unknown / course / unresolved handoff: PASS')

    load(page,'JDknown')
    page.get_by_role('button',name='Δες τη θεραπευτική πορεία').click()
    assert page.get_by_text('Τελευταία πραγματική χορήγηση: 11/04/2026').is_visible()
    assert page.locator('#doseDate').count()==0
    print('JD known actual administration reuse: PASS')

    load(page,'JDconflict')
    page.get_by_role('button',name='Δες τη θεραπευτική πορεία').click()
    assert page.get_by_text('Αντικρουόμενες καταγραφές').is_visible()
    print('JD conflict: PASS')

    load(page,'S3')
    assert page.get_by_text('Τι χρειάζεται να επανεκτιμήσουμε πριν συνεχίσουμε ή αλλάξουμε τη θεραπεία;').is_visible()
    page.get_by_role('button',name='Τι χρειάζεται να ελέγξουμε;').click()
    assert page.get_by_text('Κάταγμα παρά τη θεραπεία').is_visible()
    print('S3 fracture on treatment: PASS')

    load(page,'S4')
    assert page.get_by_text('Πώς διατηρούμε το όφελος που έχει επιτευχθεί;').is_visible()
    print('S4 post anabolic: PASS')

    load(page,'S5')
    page.get_by_role('button',name='Να εξετάσουμε τις επιλογές').click()
    assert page.get_by_text('Πρόσβαση σε θεραπεία').is_visible()
    print('S5 access distinct: PASS')

    load(page,'S6')
    page.get_by_role('button',name='Να εξετάσουμε τις επιλογές').click()
    assert page.get_by_text('Νεφρική πληροφορία').is_visible()
    print('S6 renal uncertainty: PASS')

    load(page,'S7')
    page.get_by_role('button',name='Να εξετάσουμε τις επιλογές').click()
    assert page.get_by_text('Προτίμηση ασθενούς').is_visible()
    print('S7 refusal: PASS')

    load(page,'S8')
    page.get_by_role('button',name='Να εξετάσουμε τις επιλογές').click()
    assert page.get_by_role('button',name='Άσκηση και πρόληψη πτώσεων').is_visible()
    assert page.get_by_role('button',name='Σχηματισμός οστού').count()==0
    print('S8 nonpharm: PASS')

    phone=browser.new_page(viewport={'width':390,'height':844},device_scale_factor=1)
    phone.on('pageerror',lambda error: issues.append(str(error)))
    load(phone,'JD')
    phone.screenshot(path=str(BASE/'preview-JD-mobile.png'),full_page=True)
    assert phone.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    phone.get_by_role('button',name='Δες τη θεραπευτική πορεία').click()
    assert phone.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    print('Mobile responsive: PASS')
    assert not issues,issues
    browser.close()
print('BROWSER TESTS PASS, JS errors:',issues)
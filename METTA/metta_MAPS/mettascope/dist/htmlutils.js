/** Parses a hex color string into a float array. */
export function parseHtmlColor(color) {
    return [
        parseInt(color.slice(1, 3), 16) / 255,
        parseInt(color.slice(3, 5), 16) / 255,
        parseInt(color.slice(5, 7), 16) / 255,
        1.0,
    ];
}
/** Finds an element by CSS selector in a parent element. */
export function findIn(parent, selector) {
    const elements = parent.querySelectorAll(selector);
    if (elements.length === 0) {
        throw new Error(`Element with selector "${selector}" not found`);
    }
    else if (elements.length > 1) {
        throw new Error(`Multiple elements with selector "${selector}" found`);
    }
    return elements[0];
}
/** Finds an element by CSS selector. */
export function find(selector) {
    return findIn(window.document.body, selector);
}
/** Finds multiple elements by CSS selector. */
export function finds(selector) {
    return Array.from(document.querySelectorAll(selector));
}
/** Removes all children of an element. */
export function removeChildren(element) {
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
}
/** Walks up the DOM tree and finds the given attribute. */
export function findAttr(element, attribute) {
    var e = element;
    while (e != null && e != document.body) {
        if (e.hasAttribute(attribute)) {
            return e.getAttribute(attribute);
        }
        e = e.parentElement;
    }
    return '';
}
var globalHandlers = new Map();
export function onEvent(event, selector, callback) {
    let handler = {
        selector: selector,
        event: event,
        callback: callback,
    };
    if (!globalHandlers.has(event)) {
        // This is the first time we've seen this event.
        window.addEventListener(event, (e) => {
            if (event == 'click') {
                hideMenu();
            }
            let handlers = globalHandlers.get(event);
            if (handlers) {
                var target = e.target;
                while (target != null) {
                    for (let handler of handlers) {
                        if (target.matches(handler.selector)) {
                            handler.callback(target, e);
                            return;
                        }
                    }
                    target = target.parentElement;
                }
            }
        }, { passive: false });
        globalHandlers.set(event, []);
    }
    globalHandlers.get(event)?.push(handler);
}
/**
 * Menus are hidden on any click outside the menu, and are clicked through.
 * Dropdowns are hidden on any click outside the dropdown, and are not clicked
 * through, as they have a scrim.
 */
var openMenuTarget = null;
var openMenu = null;
/** Shows a menu and sets the "scrim" target to the menu. */
export function showMenu(target, menu) {
    // Hide any other open menu.
    hideMenu();
    // Get location of the target.
    openMenuTarget = target;
    openMenu = menu;
    openMenuTarget.classList.add('selected');
    let rect = openMenuTarget.getBoundingClientRect();
    openMenu.style.left = rect.left + 'px';
    openMenu.style.top = rect.bottom + 2 + 'px';
    openMenu.classList.remove('hidden');
    // Bring the menu to the front (move it to the end of the sibling list).
    openMenu.parentElement?.appendChild(openMenu);
}
/** Hides the menu and the "scrim". */
export function hideMenu() {
    if (openMenuTarget != null) {
        openMenuTarget.classList.remove('selected');
        openMenuTarget = null;
    }
    if (openMenu != null) {
        openMenu.classList.add('hidden');
        openMenu = null;
    }
}
var openDropdownTarget = null;
var openDropdown = null;
var scrim = find('#scrim');
var scrimTarget = null;
scrim.classList.add('hidden');
onEvent('click', '#scrim', (target, event) => {
    hideMenu();
    hideDropdown();
});
/** Shows a dropdown and sets the "scrim" target to the dropdown. */
export function showDropdown(target, dropdown) {
    hideDropdown();
    openDropdown = dropdown;
    openDropdownTarget = target;
    let rect = openDropdownTarget.getBoundingClientRect();
    openDropdown.style.left = rect.left + 'px';
    openDropdown.style.top = rect.bottom + 2 + 'px';
    openDropdown.classList.remove('hidden');
    scrim.classList.remove('hidden');
}
/** Hides the dropdown and the "scrim". */
export function hideDropdown() {
    if (openDropdownTarget != null) {
        openDropdownTarget.classList.remove('selected');
        openDropdownTarget = null;
    }
    if (openDropdown != null) {
        openDropdown.classList.add('hidden');
        openDropdown = null;
    }
    scrim.classList.add('hidden');
}
/** Gets a number from local storage with a default value. */
export function localStorageGetNumber(key, defaultValue) {
    let value = localStorage.getItem(key);
    if (value == null) {
        return defaultValue;
    }
    return parseFloat(value);
}
/** Sets a number in local storage. */
export function localStorageSetNumber(key, value) {
    localStorage.setItem(key, value.toString());
}
/** Gets a whole data structure from local storage. */
export function localStorageGetObject(key, defaultValue) {
    let value = localStorage.getItem(key);
    if (value == null) {
        return defaultValue;
    }
    return JSON.parse(value);
}
/** Sets a whole data structure in local storage. */
export function localStorageSetObject(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
}
export function initHighDpiMode() {
    if (typeof window === 'undefined' || !window.document || window.devicePixelRatio <= 1) {
        return;
    }
    function swapToHighDpiImage(img) {
        let src = img.src;
        if (!src.endsWith('@2x.png') &&
            src.endsWith('.png') &&
            src.includes('data/ui/')) {
            src = src.replace(/\.png$/, '@2x.png');
            img.src = src;
        }
    }
    console.info('initHighDpiMode: All images will serve @2x.png versions.');
    document.querySelectorAll('img').forEach((img) => swapToHighDpiImage(img));
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.type === 'attributes' &&
                mutation.attributeName === 'src' &&
                mutation.target instanceof HTMLImageElement) {
                swapToHighDpiImage(mutation.target);
            }
            else if (mutation.type === 'childList') {
                for (const node of mutation.addedNodes) {
                    if (node instanceof HTMLImageElement) {
                        swapToHighDpiImage(node);
                    }
                    else if (node.nodeType === Node.ELEMENT_NODE) {
                        ;
                        node
                            .querySelectorAll('img')
                            .forEach((img) => swapToHighDpiImage(img));
                    }
                }
            }
        }
    });
    observer.observe(document.body, {
        attributes: true,
        childList: true,
        subtree: true,
        attributeFilter: ['src'],
    });
}

type Theme = "dark" | "light";

const KEY = "conscio.theme";

function read(): Theme {
  const stored = (typeof localStorage !== "undefined" && localStorage.getItem(KEY)) as Theme | null;
  if (stored === "dark" || stored === "light") return stored;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

let current = $state<Theme>(read());

export function getTheme(): Theme {
  return current;
}

export function setTheme(next: Theme): void {
  current = next;
  document.documentElement.setAttribute("data-theme", next);
  try {
    localStorage.setItem(KEY, next);
  } catch (_) {}
}

export function toggleTheme(): void {
  setTheme(current === "dark" ? "light" : "dark");
}

export const theme = {
  get value() {
    return current;
  },
  set: setTheme,
  toggle: toggleTheme,
};

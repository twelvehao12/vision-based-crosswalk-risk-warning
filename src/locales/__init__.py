from src.locales import en_us, zh_cn

DEFAULT_LOCALE = "zh_cn"

LOCALES = {
    "zh_cn": zh_cn.LOCALE,
    "en_us": en_us.LOCALE,
}

SUPPORTED_LOCALES = tuple(LOCALES)


def get_locale(name: str | None = None) -> dict:
    """Return the locale bundle for `name`, falling back to the default."""
    return LOCALES.get(name or DEFAULT_LOCALE, LOCALES[DEFAULT_LOCALE])


def t(locale: dict, table: str, key: str) -> str:
    """Look up `key` in one of the locale tables; unknown keys pass through."""
    return locale[table].get(key, key)


def format_reason(template: str, params: dict) -> str:
    """Render a reason template with raw values — used for the CSV logs."""
    return _safe_format(template, params)


def localize_reason(locale: dict, template: str, params: dict) -> str:
    """Render a reason template for display, translating the parameters."""
    localized = dict(params)
    crosswalk = localized.get("crosswalk")

    if crosswalk is not None:
        localized["crosswalk"] = t(locale, "crosswalk_name", crosswalk)

    return _safe_format(t(locale, "reason", template), localized)


def _safe_format(template: str, params: dict) -> str:
    try:
        return template.format(**params)
    except (KeyError, IndexError):
        return template

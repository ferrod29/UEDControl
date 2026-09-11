"""``python -m uedcontrol`` starts the GUI (same options as the ``uedcontrol`` command)."""

from .gui.app import main

raise SystemExit(main())

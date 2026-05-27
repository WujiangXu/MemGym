# assets/

Static images referenced by the top-level `README.md`.

## Banner slot

`README.md` reserves a hero-banner slot at the very top, commented out:

```html
<!-- ![MemGym](assets/banner.png) -->
```

Drop a `banner.png` (or `.svg`) into this directory and uncomment that line in
`README.md` to enable it. Keep the image reasonably small (≲ 200 KB) so the repo
stays clone-friendly. Until an image is added, the README renders cleanly with a
plain text title — no broken-image placeholder.

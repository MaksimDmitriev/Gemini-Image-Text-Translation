"""Provider-independent image translation pipeline."""

import io
from pathlib import Path

from PIL import Image

from drawing_translator.models import RegionProvider
from drawing_translator.rendering import apply_regions


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def translate_image(
    input_path: Path,
    output_path: Path,
    provider: RegionProvider,
    verification_passes: int = 1,
) -> None:
    with Image.open(input_path) as source:
        mime_type = Image.MIME.get(source.format, "image/png")
        image = source.convert("RGB")

    regions = provider.find_regions(input_path.read_bytes(), mime_type)
    apply_regions(image, regions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Initial pass saved {len(regions)} regions -> {output_path}")

    total_regions = len(regions)
    for pass_number in range(1, verification_passes + 1):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        remaining = provider.find_regions(
            buffer.getvalue(), "image/png", cleanup=True
        )
        apply_regions(image, remaining)
        total_regions += len(remaining)
        image.save(output_path)
        print(
            f"Verification pass {pass_number} saved {len(remaining)} regions "
            f"-> {output_path}"
        )
        if not remaining:
            break
    print(f"Translated {total_regions} regions with {provider.name} -> {output_path}")


def translate_path(
    source: Path,
    output: Path | None,
    provider: RegionProvider,
    verification_passes: int = 1,
) -> None:
    if source.is_dir():
        images = sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"no PNG, JPEG, or WebP images found in {source}")
        output_folder = output or Path("output")
        for index, image_path in enumerate(images, start=1):
            destination = output_folder / f"{image_path.stem}_en.png"
            print(f"Processing {index}/{len(images)}: {image_path}")
            translate_image(
                image_path, destination, provider, verification_passes
            )
        return

    if source.is_file():
        destination = output or source.with_name(f"{source.stem}_translated.png")
        translate_image(source, destination, provider, verification_passes)
        return

    raise ValueError(f"input does not exist: {source}")

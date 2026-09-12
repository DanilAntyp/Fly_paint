"""Local, bounded sketch preprocessing. No tracing path or action labels."""
import base64
from io import BytesIO
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from scipy.ndimage import maximum_filter, minimum_filter

PAPER = (246, 241, 225)
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000


def png_url(image):
    stream = BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


def raster_image(target):
    pixels = np.full((*target.shape, 3), PAPER, dtype=np.uint8)
    pixels[target] = (25, 28, 23)
    return Image.fromarray(pixels)


def decode_image(data):
    if len(data) > MAX_BYTES:
        raise ValueError('Image exceeds the 10 MB limit.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {'PNG', 'JPEG', 'WEBP'}:
                    raise ValueError('Choose a PNG, JPEG or WebP image.')
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError('Image exceeds the 16 megapixel limit.')
                source.load()
                image = ImageOps.exif_transpose(source).convert('RGBA')
                paper = Image.new('RGBA', image.size, PAPER + (255,))
                return Image.alpha_composite(paper, image).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError('Cannot decode image safely. Choose a valid PNG, JPEG or WebP.') from error


def prepare(image, size=128, threshold=35):
    if size not in (64, 128, 256) or not 5 <= threshold <= 200:
        raise ValueError('Invalid target size or detail threshold.')
    fitted = ImageOps.contain(image, (size - 16, size - 16), Image.Resampling.LANCZOS)
    paper = Image.new('RGB', (size, size), PAPER)
    paper.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    gray = np.asarray(ImageOps.grayscale(paper), dtype=np.int16)
    # Morphological contrast outlines both dark and light objects, independent of RGB.
    contrast = maximum_filter(gray, 3) - minimum_filter(gray, 3)
    target = contrast >= threshold
    target[:4] = target[-4:] = False
    target[:, :4] = target[:, -4:] = False
    return target


def fixture(name, size=128):
    image = Image.new('RGB', (size, size), PAPER)
    d = ImageDraw.Draw(image)
    k = size / 128
    def points(coords):
        return [(x*k, y*k) for x, y in coords]
    width = max(1, round(2*k))
    if name == 'circle':
        d.ellipse((30*k, 30*k, 98*k, 98*k), outline='black', width=width)
    elif name == 'square':
        d.rectangle((30*k, 30*k, 98*k, 98*k), outline='black', width=width)
    elif name == 'spiral':
        t = np.linspace(0, 4*np.pi, 300)
        d.line([(64*k+(5+2.7*a)*k*np.cos(a), 64*k+(5+2.7*a)*k*np.sin(a)) for a in t], fill='black', width=width)
    elif name == 'cat':
        d.line(points([(26,90),(26,36),(44,48),(64,42),(84,48),(102,36),(102,90),(82,104),(46,104),(26,90)]), fill='black', width=width)
        for x in (46, 80):
            d.ellipse((x*k,65*k,(x+4)*k,69*k), fill='black')
        d.line(points([(57,81),(64,85),(71,81)]), fill='black', width=width)
    elif name == 'leaf':
        d.line(points([(28,104),(32,57),(61,27),(102,22),(98,65),(66,96),(28,104),(83,42)]), fill='black', width=width)
    elif name != 'free':
        raise ValueError('Unknown preset.')
    return image

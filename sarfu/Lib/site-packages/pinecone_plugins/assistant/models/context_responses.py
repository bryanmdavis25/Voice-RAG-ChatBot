from dataclasses import dataclass
from typing import TypeVar, Union, Optional

from pinecone_plugins.assistant.data.core.client.model.context_model import (
    ContextModel as OpenAPIContextModel,
)
from pinecone_plugins.assistant.models.core.dataclass import BaseDataclass
from pinecone_plugins.assistant.models.file_model import FileModel
from pinecone_plugins.assistant.models.shared import TokenCounts

RefType = TypeVar(
    "RefType",
    bound=Union["TextReference", "PdfReference", "MarkdownReference", "JsonReference"],
)


SnippetType = TypeVar(
    "SnippetType",
    bound=Union["TextSnippet", "MultimodalSnippet"],
)


MultimodalContentBlockType = TypeVar(
    "MultimodalContentBlockType",
    bound=Union["TextBlock", "ImageBlock"],
)


# ---------- Reference Models ----------


@dataclass
class BaseReference(BaseDataclass):
    type: str

    @classmethod
    def from_openapi(cls, value):
        raise NotImplementedError


@dataclass
class PdfReference(BaseReference):
    pages: list[int]
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict) -> "PdfReference":
        return cls(
            type=ref_dict["type"],
            pages=ref_dict["pages"],
            file=FileModel.from_dict(ref_dict["file"]),
        )


@dataclass
class DocxReference(BaseReference):
    pages: list[int]
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict) -> "PdfReference":
        return cls(
            type=ref_dict["type"],
            pages=ref_dict["pages"],
            file=FileModel.from_dict(ref_dict["file"]),
        )


@dataclass
class TextReference(BaseReference):
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict) -> "TextReference":
        return cls(type=ref_dict["type"], file=FileModel.from_dict(ref_dict["file"]))


@dataclass
class MarkdownReference(BaseReference):
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict) -> "MarkdownReference":
        return cls(type=ref_dict["type"], file=FileModel.from_dict(ref_dict["file"]))


@dataclass
class JsonReference(BaseReference):
    file: FileModel

    @classmethod
    def from_openapi(cls, ref_dict: dict) -> "JsonReference":
        return cls(type=ref_dict["type"], file=FileModel.from_dict(ref_dict["file"]))


class TypedReference:
    @classmethod
    def from_openapi(cls, d: dict) -> RefType:
        type_ = d["type"]
        ref_map = {
            "text": TextReference,
            "doc_x": DocxReference,
            "pdf": PdfReference,
            "markdown": MarkdownReference,
            "json": JsonReference,
        }
        return ref_map[type_].from_openapi(d)


# ---------- Multimodal content blocks ----------


@dataclass
class BaseMultimodalContentBlock(BaseDataclass):
    type: str

    @classmethod
    def from_openapi(cls, value):
        raise NotImplementedError
    

@dataclass
class TextBlock(BaseMultimodalContentBlock):
    text: str

    @classmethod
    def from_openapi(cls, d: dict) -> "TextBlock":
        return cls(
            type=d["type"],
            text=d["text"],
        )


@dataclass
class Image(BaseDataclass):
    mime_type: str
    data: str
    type: str

    @classmethod
    def from_openapi(cls, d: dict):
        return cls(
            mime_type=d["mime_type"],
            data=d["data"],
            type=d["type"],
        )

@dataclass
class ImageBlock(BaseMultimodalContentBlock):
    caption: str
    image: Optional[Image]

    @classmethod
    def from_openapi(cls, d: dict) -> "ImageBlock":
        image_info = d.get("image")
        if image_info:
            return cls(
                type=d["type"],
                caption=d["caption"],
                image=Image.from_openapi(image_info),
            )
        else:
            return cls(
                type=d["type"],
                caption=d["caption"],
                image=None,
            )


# ---------- Snippet Models ----------


@dataclass
class BaseSnippet(BaseDataclass):
    type: str

    @classmethod
    def from_openapi(cls, value):
        raise NotImplementedError


@dataclass
class TextSnippet(BaseSnippet):
    content: str
    score: float
    reference: RefType

    @classmethod
    def from_openapi(cls, d: dict) -> "TextSnippet":
        return cls(
            type=d["type"],
            content=d["content"],
            score=d["score"],
            reference=TypedReference.from_openapi(d["reference"]),
        )
    

@dataclass
class MultimodalSnippet(BaseSnippet):
    content: list[MultimodalContentBlockType]
    score: float
    reference: RefType

    @classmethod
    def from_openapi(cls, d: dict) -> "MultimodalSnippet":
        blocks: list[MultimodalContentBlockType] = []
        for block in d["content"]:
            type_ = block["type"]
            block_map = {
                "text": TextBlock,
                "image": ImageBlock,
            }
            blocks.append(block_map[type_].from_openapi(block))

        return cls(
            type=d["type"],
            content=blocks,
            score=d["score"],
            reference=TypedReference.from_openapi(d["reference"]),
        )


@dataclass
class Snippet:
    @classmethod
    def from_openapi(cls, snippet: dict) -> SnippetType:
        type_ = snippet["type"]
        sinnpet_map = {
            "text": TextSnippet,
            "multimodal": MultimodalSnippet,
        }
        return sinnpet_map[type_].from_openapi(snippet)


@dataclass
class ContextResponse(BaseDataclass):
    id: str
    snippets: list[Snippet]
    usage: TokenCounts

    @classmethod
    def from_openapi(cls, ctx: OpenAPIContextModel):
        return cls(
            id=ctx.id,
            snippets=[Snippet.from_openapi(snippet) for snippet in ctx.snippets],
            usage=TokenCounts.from_openapi(ctx.usage),
        )

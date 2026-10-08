"""Tests for UPnP metadata parsing helpers."""

from __future__ import annotations

from pywiim.upnp.metadata import is_valid_image_url, parse_didl_metadata, parse_getinfoex_response

DIDL_ESCAPED = (
    "&lt;?xml version=&quot;1.0&quot; encoding=&quot;UTF-8&quot;?&gt;"
    "&lt;DIDL-Lite xmlns:dc=&quot;http://purl.org/dc/elements/1.1/&quot; "
    "xmlns:upnp=&quot;urn:schemas-upnp-org:metadata-1-0/upnp/&quot; "
    "xmlns=&quot;urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/&quot;&gt;"
    "&lt;item id=&quot;0&quot;&gt;"
    "&lt;dc:title&gt;Herz an Herz&lt;/dc:title&gt;"
    "&lt;upnp:artist&gt;Betontod&lt;/upnp:artist&gt;"
    "&lt;upnp:album&gt;Revolution&lt;/upnp:album&gt;"
    "&lt;upnp:albumArtURI&gt;https://i.scdn.co/image/ab67616d0000b27356d04017e7964e8b231a5677&lt;/upnp:albumArtURI&gt;"
    "&lt;/item&gt;&lt;/DIDL-Lite&gt;"
)

SOAP_RESPONSE = f"""<?xml version=\"1.0\" encoding=\"utf-8\"?>
<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\">
  <s:Body>
    <u:GetInfoExResponse xmlns:u=\"urn:schemas-upnp-org:service:AVTransport:1\">
      <CurrentTransportState>PLAYING</CurrentTransportState>
      <TrackMetaData>{DIDL_ESCAPED}</TrackMetaData>
      <PlayMedium>SPOTIFY</PlayMedium>
    </u:GetInfoExResponse>
  </s:Body>
</s:Envelope>"""


class TestUpnpMetadataHelpers:
    def test_is_valid_image_url(self):
        assert is_valid_image_url("https://example.com/art.jpg") is True
        assert is_valid_image_url("un_known") is False
        assert is_valid_image_url(None) is False

    def test_parse_didl_metadata_extracts_artwork(self):
        result = parse_didl_metadata(DIDL_ESCAPED)
        assert result["title"] == "Herz an Herz"
        assert result["artist"] == "Betontod"
        assert result["album"] == "Revolution"
        assert result["image_url"].startswith("https://i.scdn.co/image/")

    def test_parse_didl_metadata_sanitizes_bare_ampersands(self):
        didl = (
            '<DIDL-Lite xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
            "<item><dc:title>A & B</dc:title>"
            "<upnp:albumArtURI>https://example.com/a.jpg?x=1&y=2</upnp:albumArtURI>"
            "</item></DIDL-Lite>"
        )

        result = parse_didl_metadata(didl)

        assert result["title"] == "A & B"
        assert result["image_url"] == "https://example.com/a.jpg?x=1&y=2"

    def test_parse_didl_metadata_invalid_artwork(self):
        didl = (
            '<DIDL-Lite xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
            "<item><upnp:albumArtURI>un_known</upnp:albumArtURI></item></DIDL-Lite>"
        )
        result = parse_didl_metadata(didl)
        assert "image_url" not in result

    def test_parse_getinfoex_response(self):
        result = parse_getinfoex_response(SOAP_RESPONSE)
        assert result["CurrentTransportState"] == "PLAYING"
        assert result["PlayMedium"] == "SPOTIFY"
        assert result["title"] == "Herz an Herz"
        assert result["image_url"].startswith("https://i.scdn.co/image/")

    def test_parse_didl_metadata_extracts_input_codec(self):
        # Fixed-input (HDMI) DIDL uses the www.wiimu.com vendor namespace and
        # carries the codec in song:coding_f. Matched by local name so the
        # namespace URI variant does not matter. Normalized to lowercase.
        didl = (
            '<DIDL-Lite xmlns:song="www.wiimu.com/song/" '
            'xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/">'
            '<item id="0"><song:coding_f>AC3</song:coding_f>'
            "<res>HDMI</res></item></DIDL-Lite>"
        )
        assert parse_didl_metadata(didl)["codec"] == "ac3"

    def test_parse_didl_metadata_codec_absent_and_placeholder(self):
        # No coding_f element -> no codec key.
        no_codec = (
            '<DIDL-Lite xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/">'
            "<item><dc:title>x</dc:title></item></DIDL-Lite>"
        )
        assert "codec" not in parse_didl_metadata(no_codec)

        # Empty coding_f with allow_clear clears the field.
        empty = (
            '<DIDL-Lite xmlns:song="www.wiimu.com/song/" '
            'xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/">'
            "<item><song:coding_f></song:coding_f></item></DIDL-Lite>"
        )
        assert parse_didl_metadata(empty, allow_clear=True)["codec"] is None

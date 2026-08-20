use super::Crc32;

fn require() -> Option<Crc32> {
	Crc32::detect()
}

/// Reflected CRC-32 byte step (standard zlib/Ethernet algorithm: ARM's
/// `CRC32*`/`CRC32C*` implement exactly this, one byte at a time, folding
/// multi-byte operands in little-endian byte order).
fn crc_byte(crc: u32, byte: u8, poly: u32) -> u32 {
	let mut c = crc ^ byte as u32;
	for _ in 0..8 {
		c = if c & 1 != 0 { (c >> 1) ^ poly } else { c >> 1 };
	}
	c
}

fn crc_bytes(crc: u32, bytes: &[u8], poly: u32) -> u32 {
	bytes.iter().fold(crc, |c, &b| crc_byte(c, b, poly))
}

const CRC32_POLY: u32 = 0xEDB8_8320;
const CRC32C_POLY: u32 = 0x82F6_3B78;

#[test]
fn crc32_family_matches_reflected_reference() {
	let Some(t) = require() else { return };
	for crc in [0u32, 1, 0xFFFF_FFFF, 0x1234_5678] {
		assert_eq!(t.crc32b(crc, 0xAB), crc_bytes(crc, &[0xAB], CRC32_POLY), "crc32b({crc:#x})");
		assert_eq!(t.crc32h(crc, 0xABCD), crc_bytes(crc, &0xABCDu16.to_le_bytes(), CRC32_POLY), "crc32h({crc:#x})");
		assert_eq!(
			t.crc32w(crc, 0xDEAD_BEEF),
			crc_bytes(crc, &0xDEAD_BEEFu32.to_le_bytes(), CRC32_POLY),
			"crc32w({crc:#x})"
		);
		assert_eq!(
			t.crc32d(crc, 0x0123_4567_89AB_CDEF),
			crc_bytes(crc, &0x0123_4567_89AB_CDEFu64.to_le_bytes(), CRC32_POLY),
			"crc32d({crc:#x})"
		);
	}
}

#[test]
fn crc32c_family_matches_reflected_reference() {
	let Some(t) = require() else { return };
	for crc in [0u32, 1, 0xFFFF_FFFF, 0x1234_5678] {
		assert_eq!(t.crc32cb(crc, 0xAB), crc_bytes(crc, &[0xAB], CRC32C_POLY), "crc32cb({crc:#x})");
		assert_eq!(t.crc32ch(crc, 0xABCD), crc_bytes(crc, &0xABCDu16.to_le_bytes(), CRC32C_POLY), "crc32ch({crc:#x})");
		assert_eq!(
			t.crc32cw(crc, 0xDEAD_BEEF),
			crc_bytes(crc, &0xDEAD_BEEFu32.to_le_bytes(), CRC32C_POLY),
			"crc32cw({crc:#x})"
		);
		assert_eq!(
			t.crc32cd(crc, 0x0123_4567_89AB_CDEF),
			crc_bytes(crc, &0x0123_4567_89AB_CDEFu64.to_le_bytes(), CRC32C_POLY),
			"crc32cd({crc:#x})"
		);
	}
}

#[test]
fn crc32b_chained_matches_standard_crc32_of_bytes() {
	let Some(t) = require() else { return };
	// Standard CRC-32/ISO-HDLC check value for the ASCII string "123456789"
	// is 0xCBF43926 (well-known test vector, e.g. used by the Rocksoft CRC
	// catalogue) with init 0xFFFFFFFF and a final XOR of 0xFFFFFFFF.
	let data = b"123456789";
	let mut crc = 0xFFFF_FFFFu32;
	for &b in data {
		crc = t.crc32b(crc, b);
	}
	assert_eq!(crc ^ 0xFFFF_FFFF, 0xCBF4_3926);
}

#[test]
fn crc32cb_chained_matches_standard_crc32c_of_bytes() {
	let Some(t) = require() else { return };
	// Standard CRC-32C/ISCSI check value for "123456789" is 0xE3069283
	// (same init/final-XOR convention as the CRC-32 vector above).
	let data = b"123456789";
	let mut crc = 0xFFFF_FFFFu32;
	for &b in data {
		crc = t.crc32cb(crc, b);
	}
	assert_eq!(crc ^ 0xFFFF_FFFF, 0xE306_9283);
}

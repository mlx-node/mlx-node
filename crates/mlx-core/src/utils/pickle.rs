//! Minimal Python Pickle Deserializer
//!
//! Handles pickle protocols 0-5 as used by PyTorch `torch.save()` and
//! Paddle `paddle.save()`. Only the opcodes actually emitted by these
//! frameworks are implemented; unsupported opcodes return an error.
//!
//! The parser produces a tree of `PickleValue` nodes. Higher-level
//! interpreters (PyTorch loader, numpy/Paddle loader) walk this tree
//! to extract tensor metadata and raw data.

use std::collections::HashMap;
use std::io::{Cursor, Read as IoRead};

use napi::bindgen_prelude::*;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Bytes(Vec<u8>),
    String(String),
    List(Vec<PickleValue>),
    Tuple(Vec<PickleValue>),
    Dict(Vec<(PickleValue, PickleValue)>),
    /// A reconstructed Python object: callable(*args) with optional __setstate__
    Object {
        module: String,
        name: String,
        args: Vec<PickleValue>,
        state: Option<Box<PickleValue>>,
    },
    /// A raw GLOBAL reference (not yet REDUCEd)
    Global {
        module: String,
        name: String,
    },
}

impl PickleValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            PickleValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            PickleValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            PickleValue::Bytes(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_tuple(&self) -> Option<&[PickleValue]> {
        match self {
            PickleValue::Tuple(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[PickleValue]> {
        match self {
            PickleValue::List(l) => Some(l),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&[(PickleValue, PickleValue)]> {
        match self {
            PickleValue::Dict(d) => Some(d),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<(&str, &str, &[PickleValue], Option<&PickleValue>)> {
        match self {
            PickleValue::Object {
                module,
                name,
                args,
                state,
            } => Some((module, name, args, state.as_deref())),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pickle VM
// ---------------------------------------------------------------------------

/// Sentinel value pushed by MARK opcode. We use a separate enum to
/// distinguish marks from real values on the stack.
enum StackItem {
    Value(PickleValue),
    Mark,
}

pub(crate) fn unpickle(data: &[u8]) -> Result<PickleValue> {
    let mut cur = Cursor::new(data);
    let mut stack: Vec<StackItem> = Vec::new();
    let mut memo: HashMap<u32, PickleValue> = HashMap::new();

    loop {
        let op = read_u8(&mut cur)?;

        match op {
            // -- Protocol / framing --
            0x80 => {
                // PROTO
                let _version = read_u8(&mut cur)?;
            }
            0x95 => {
                // FRAME (proto 4)
                let _len = read_u64(&mut cur)?;
            }

            // -- Constants --
            b'N' => stack.push(StackItem::Value(PickleValue::None)),
            0x88 => stack.push(StackItem::Value(PickleValue::Bool(true))), // NEWTRUE
            0x89 => stack.push(StackItem::Value(PickleValue::Bool(false))), // NEWFALSE

            // -- Integers --
            b'K' => {
                // BININT1
                let v = read_u8(&mut cur)? as i64;
                stack.push(StackItem::Value(PickleValue::Int(v)));
            }
            b'M' => {
                // BININT2
                let v = read_u16(&mut cur)? as i64;
                stack.push(StackItem::Value(PickleValue::Int(v)));
            }
            b'J' => {
                // BININT (signed 32-bit)
                let v = read_i32(&mut cur)? as i64;
                stack.push(StackItem::Value(PickleValue::Int(v)));
            }
            0x8a => {
                // LONG1 (1-byte length, little-endian signed integer)
                let n = read_u8(&mut cur)? as usize;
                if n > 8 {
                    return Err(Error::from_reason(format!(
                        "pickle LONG1 too large for i64: {n} bytes"
                    )));
                }
                let mut buf = vec![0u8; n];
                cur.read_exact(&mut buf)
                    .map_err(|e| Error::from_reason(format!("pickle LONG1 read: {e}")))?;
                let v = long_from_bytes(&buf);
                stack.push(StackItem::Value(PickleValue::Int(v)));
            }
            b'I' => {
                // INT (text)
                let line = read_line(&mut cur)?;
                let trimmed = line.trim();
                if trimmed == "01" {
                    stack.push(StackItem::Value(PickleValue::Bool(true)));
                } else if trimmed == "00" {
                    stack.push(StackItem::Value(PickleValue::Bool(false)));
                } else {
                    let v: i64 = trimmed
                        .parse()
                        .map_err(|e| Error::from_reason(format!("pickle INT parse: {e}")))?;
                    stack.push(StackItem::Value(PickleValue::Int(v)));
                }
            }

            // -- Float --
            b'G' => {
                // BINFLOAT (big-endian f64)
                let mut buf = [0u8; 8];
                cur.read_exact(&mut buf)
                    .map_err(|e| Error::from_reason(format!("pickle BINFLOAT read: {e}")))?;
                let v = f64::from_be_bytes(buf);
                stack.push(StackItem::Value(PickleValue::Float(v)));
            }

            // -- Bytes --
            b'B' => {
                // BINBYTES (4-byte length)
                let len = read_u32(&mut cur)? as usize;
                let data = read_bytes(&mut cur, len)?;
                stack.push(StackItem::Value(PickleValue::Bytes(data)));
            }
            b'C' => {
                // SHORT_BINBYTES (1-byte length)
                let len = read_u8(&mut cur)? as usize;
                let data = read_bytes(&mut cur, len)?;
                stack.push(StackItem::Value(PickleValue::Bytes(data)));
            }
            0x8e => {
                // BINBYTES8 (8-byte length)
                let len = read_u64(&mut cur)? as usize;
                let data = read_bytes(&mut cur, len)?;
                stack.push(StackItem::Value(PickleValue::Bytes(data)));
            }

            // -- Strings --
            0x8c => {
                // SHORT_BINUNICODE (1-byte length)
                let len = read_u8(&mut cur)? as usize;
                let data = read_bytes(&mut cur, len)?;
                let s = String::from_utf8(data)
                    .map_err(|e| Error::from_reason(format!("pickle string utf8: {e}")))?;
                stack.push(StackItem::Value(PickleValue::String(s)));
            }
            b'X' => {
                // BINUNICODE (4-byte length)
                let len = read_u32(&mut cur)? as usize;
                let data = read_bytes(&mut cur, len)?;
                let s = String::from_utf8(data)
                    .map_err(|e| Error::from_reason(format!("pickle string utf8: {e}")))?;
                stack.push(StackItem::Value(PickleValue::String(s)));
            }
            b'T' => {
                // BINSTRING (4-byte length)
                let len = read_u32(&mut cur)? as usize;
                let data = read_bytes(&mut cur, len)?;
                let s = String::from_utf8_lossy(&data).into_owned();
                stack.push(StackItem::Value(PickleValue::String(s)));
            }
            b'U' => {
                // SHORT_BINSTRING (1-byte length)
                let len = read_u8(&mut cur)? as usize;
                let data = read_bytes(&mut cur, len)?;
                let s = String::from_utf8_lossy(&data).into_owned();
                stack.push(StackItem::Value(PickleValue::String(s)));
            }

            // -- Mark & Tuple --
            b'(' => stack.push(StackItem::Mark),
            b')' => stack.push(StackItem::Value(PickleValue::Tuple(vec![]))),
            b't' => {
                // TUPLE: pop items until MARK
                let items = pop_mark(&mut stack)?;
                stack.push(StackItem::Value(PickleValue::Tuple(items)));
            }
            0x85 => {
                // TUPLE1
                let a = pop_value(&mut stack)?;
                stack.push(StackItem::Value(PickleValue::Tuple(vec![a])));
            }
            0x86 => {
                // TUPLE2
                let b = pop_value(&mut stack)?;
                let a = pop_value(&mut stack)?;
                stack.push(StackItem::Value(PickleValue::Tuple(vec![a, b])));
            }
            0x87 => {
                // TUPLE3
                let c = pop_value(&mut stack)?;
                let b = pop_value(&mut stack)?;
                let a = pop_value(&mut stack)?;
                stack.push(StackItem::Value(PickleValue::Tuple(vec![a, b, c])));
            }

            // -- List --
            b']' => stack.push(StackItem::Value(PickleValue::List(vec![]))),
            b'l' => {
                let items = pop_mark(&mut stack)?;
                stack.push(StackItem::Value(PickleValue::List(items)));
            }
            b'a' => {
                // APPEND
                let item = pop_value(&mut stack)?;
                match stack.last_mut() {
                    Some(StackItem::Value(PickleValue::List(list))) => list.push(item),
                    _ => {
                        return Err(Error::from_reason(
                            "pickle APPEND: top is not a list".to_string(),
                        ));
                    }
                }
            }
            b'e' => {
                // APPENDS
                let items = pop_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(StackItem::Value(PickleValue::List(list))) => list.extend(items),
                    _ => {
                        return Err(Error::from_reason(
                            "pickle APPENDS: top is not a list".to_string(),
                        ));
                    }
                }
            }

            // -- Dict --
            b'}' => stack.push(StackItem::Value(PickleValue::Dict(vec![]))),
            b'd' => {
                let items = pop_mark(&mut stack)?;
                let pairs = items_to_pairs(items)?;
                stack.push(StackItem::Value(PickleValue::Dict(pairs)));
            }
            b's' => {
                // SETITEM
                let val = pop_value(&mut stack)?;
                let key = pop_value(&mut stack)?;
                match stack.last_mut() {
                    Some(StackItem::Value(PickleValue::Dict(d))) => d.push((key, val)),
                    _ => {
                        return Err(Error::from_reason(
                            "pickle SETITEM: top is not a dict".to_string(),
                        ));
                    }
                }
            }
            b'u' => {
                // SETITEMS
                let items = pop_mark(&mut stack)?;
                let pairs = items_to_pairs(items)?;
                match stack.last_mut() {
                    Some(StackItem::Value(PickleValue::Dict(d))) => d.extend(pairs),
                    Some(StackItem::Value(PickleValue::Object { name, .. }))
                        if name == "OrderedDict" =>
                    {
                        // OrderedDict constructed via REDUCE then populated with SETITEMS.
                        // Replace the Object with a plain Dict.
                        *stack.last_mut().unwrap() = StackItem::Value(PickleValue::Dict(pairs));
                    }
                    _ => {
                        return Err(Error::from_reason(
                            "pickle SETITEMS: top is not a dict".to_string(),
                        ));
                    }
                }
            }

            // -- Object construction --
            b'c' => {
                // GLOBAL (text: module\nname\n)
                let module = read_line(&mut cur)?;
                let name = read_line(&mut cur)?;
                stack.push(StackItem::Value(PickleValue::Global {
                    module: module.trim().to_string(),
                    name: name.trim().to_string(),
                }));
            }
            0x93 => {
                // STACK_GLOBAL (proto 4): pop name, pop module
                let name = pop_value(&mut stack)?;
                let module = pop_value(&mut stack)?;
                let module_str = match &module {
                    PickleValue::String(s) => s.clone(),
                    _ => {
                        return Err(Error::from_reason(
                            "STACK_GLOBAL: module is not string".to_string(),
                        ));
                    }
                };
                let name_str = match &name {
                    PickleValue::String(s) => s.clone(),
                    _ => {
                        return Err(Error::from_reason(
                            "STACK_GLOBAL: name is not string".to_string(),
                        ));
                    }
                };
                stack.push(StackItem::Value(PickleValue::Global {
                    module: module_str,
                    name: name_str,
                }));
            }
            b'R' => {
                // REDUCE: pop args, pop callable, push callable(*args)
                let args_val = pop_value(&mut stack)?;
                let callable = pop_value(&mut stack)?;
                match callable {
                    PickleValue::Global { module, name } => {
                        let args = match args_val {
                            PickleValue::Tuple(t) => t,
                            other => vec![other],
                        };
                        stack.push(StackItem::Value(PickleValue::Object {
                            module,
                            name,
                            args,
                            state: None,
                        }));
                    }
                    _ => {
                        // Callable is already a reconstructed object (e.g., nested REDUCE)
                        // Treat as opaque object
                        stack.push(StackItem::Value(PickleValue::Object {
                            module: "__reduce__".to_string(),
                            name: "unknown".to_string(),
                            args: vec![callable, args_val],
                            state: None,
                        }));
                    }
                }
            }
            0x81 => {
                // NEWOBJ: pop args, pop cls, push cls.__new__(cls, *args)
                let args_val = pop_value(&mut stack)?;
                let cls = pop_value(&mut stack)?;
                match cls {
                    PickleValue::Global { module, name } => {
                        let args = match args_val {
                            PickleValue::Tuple(t) => t,
                            other => vec![other],
                        };
                        stack.push(StackItem::Value(PickleValue::Object {
                            module,
                            name,
                            args,
                            state: None,
                        }));
                    }
                    _ => {
                        stack.push(StackItem::Value(PickleValue::Object {
                            module: "__newobj__".to_string(),
                            name: "unknown".to_string(),
                            args: vec![cls, args_val],
                            state: None,
                        }));
                    }
                }
            }
            0x92 => {
                // NEWOBJ_EX (proto 4): pop kwargs, pop args, pop cls
                let _kwargs = pop_value(&mut stack)?;
                let args_val = pop_value(&mut stack)?;
                let cls = pop_value(&mut stack)?;
                match cls {
                    PickleValue::Global { module, name } => {
                        let args = match args_val {
                            PickleValue::Tuple(t) => t,
                            other => vec![other],
                        };
                        stack.push(StackItem::Value(PickleValue::Object {
                            module,
                            name,
                            args,
                            state: None,
                        }));
                    }
                    _ => {
                        stack.push(StackItem::Value(PickleValue::Object {
                            module: "__newobj_ex__".to_string(),
                            name: "unknown".to_string(),
                            args: vec![cls, args_val],
                            state: None,
                        }));
                    }
                }
            }
            b'b' => {
                // BUILD: pop state, apply to top
                let state = pop_value(&mut stack)?;
                match stack.last_mut() {
                    Some(StackItem::Value(PickleValue::Object { state: s, .. })) => {
                        *s = Some(Box::new(state));
                    }
                    Some(StackItem::Value(PickleValue::Dict(d))) => {
                        // BUILD on a dict: merge state dict into it
                        if let PickleValue::Dict(pairs) = state {
                            d.extend(pairs);
                        }
                    }
                    _ => {
                        // Ignore BUILD on non-objects (some pickle streams do this)
                    }
                }
            }

            // -- Memo --
            b'p' => {
                // PUT (text key)
                let key_str = read_line(&mut cur)?;
                let key: u32 = key_str
                    .trim()
                    .parse()
                    .map_err(|e| Error::from_reason(format!("pickle PUT parse: {e}")))?;
                if let Some(StackItem::Value(val)) = stack.last() {
                    memo.insert(key, val.clone());
                }
            }
            b'q' => {
                // BINPUT (1-byte key)
                let key = read_u8(&mut cur)? as u32;
                if let Some(StackItem::Value(val)) = stack.last() {
                    memo.insert(key, val.clone());
                }
            }
            b'r' => {
                // LONG_BINPUT (4-byte key)
                let key = read_u32(&mut cur)?;
                if let Some(StackItem::Value(val)) = stack.last() {
                    memo.insert(key, val.clone());
                }
            }
            0x94 => {
                // MEMOIZE (proto 4): store top in next memo slot
                let key = memo.len() as u32;
                if let Some(StackItem::Value(val)) = stack.last() {
                    memo.insert(key, val.clone());
                }
            }
            b'g' => {
                // GET (text key)
                let key_str = read_line(&mut cur)?;
                let key: u32 = key_str
                    .trim()
                    .parse()
                    .map_err(|e| Error::from_reason(format!("pickle GET parse: {e}")))?;
                let val = memo.get(&key).cloned().ok_or_else(|| {
                    Error::from_reason(format!("pickle GET: key {key} not found"))
                })?;
                stack.push(StackItem::Value(val));
            }
            b'h' => {
                // BINGET (1-byte key)
                let key = read_u8(&mut cur)? as u32;
                let val = memo.get(&key).cloned().ok_or_else(|| {
                    Error::from_reason(format!("pickle BINGET: key {key} not found"))
                })?;
                stack.push(StackItem::Value(val));
            }
            b'j' => {
                // LONG_BINGET (4-byte key)
                let key = read_u32(&mut cur)?;
                let val = memo.get(&key).cloned().ok_or_else(|| {
                    Error::from_reason(format!("pickle LONG_BINGET: key {key} not found"))
                })?;
                stack.push(StackItem::Value(val));
            }

            // -- Persistent ID (used by PyTorch for tensor storage references) --
            b'Q' => {
                // BINPERSID: pop persistent_id, push persistent_load(pid)
                // PyTorch pid format: ('storage', StorageClass, key, location, numel)
                let pid = pop_value(&mut stack)?;
                match pid {
                    PickleValue::Tuple(ref items) if items.len() >= 3 => {
                        // Extract storage class and key from the tuple
                        // items[0] = 'storage' string
                        // items[1] = StorageClass (Global)
                        // items[2] = storage key (string)
                        // items[3] = location (string, e.g. 'cpu')
                        // items[4] = num_elements (int)
                        stack.push(StackItem::Value(PickleValue::Object {
                            module: "torch".to_string(),
                            name: "storage".to_string(),
                            args: items.clone(),
                            state: None,
                        }));
                    }
                    _ => {
                        // Unknown persistent ID format, push as-is wrapped in Object
                        stack.push(StackItem::Value(PickleValue::Object {
                            module: "torch".to_string(),
                            name: "persistent_load".to_string(),
                            args: vec![pid],
                            state: None,
                        }));
                    }
                }
            }

            // -- Stack manipulation --
            b'0' => {
                // POP
                stack.pop();
            }
            b'1' => {
                // POP_MARK
                pop_mark(&mut stack)?;
            }
            b'2' => {
                // DUP
                if let Some(StackItem::Value(val)) = stack.last() {
                    let cloned = val.clone();
                    stack.push(StackItem::Value(cloned));
                }
            }

            // -- Stop --
            b'.' => {
                return pop_value(&mut stack);
            }

            _ => {
                return Err(Error::from_reason(format!(
                    "pickle: unsupported opcode 0x{op:02x} at offset {}",
                    cur.position() - 1
                )));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_u8(cur: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut buf = [0u8; 1];
    cur.read_exact(&mut buf)
        .map_err(|e| Error::from_reason(format!("pickle read_u8: {e}")))?;
    Ok(buf[0])
}

fn read_u16(cur: &mut Cursor<&[u8]>) -> Result<u16> {
    let mut buf = [0u8; 2];
    cur.read_exact(&mut buf)
        .map_err(|e| Error::from_reason(format!("pickle read_u16: {e}")))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i32(cur: &mut Cursor<&[u8]>) -> Result<i32> {
    let mut buf = [0u8; 4];
    cur.read_exact(&mut buf)
        .map_err(|e| Error::from_reason(format!("pickle read_i32: {e}")))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u32(cur: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cur.read_exact(&mut buf)
        .map_err(|e| Error::from_reason(format!("pickle read_u32: {e}")))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(cur: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut buf = [0u8; 8];
    cur.read_exact(&mut buf)
        .map_err(|e| Error::from_reason(format!("pickle read_u64: {e}")))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_bytes(cur: &mut Cursor<&[u8]>, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    cur.read_exact(&mut buf)
        .map_err(|e| Error::from_reason(format!("pickle read_bytes({len}): {e}")))?;
    Ok(buf)
}

fn read_line(cur: &mut Cursor<&[u8]>) -> Result<String> {
    let mut buf = Vec::new();
    loop {
        let b = read_u8(cur)?;
        if b == b'\n' {
            break;
        }
        buf.push(b);
    }
    String::from_utf8(buf).map_err(|e| Error::from_reason(format!("pickle read_line utf8: {e}")))
}

/// Convert little-endian signed integer bytes to i64
fn long_from_bytes(bytes: &[u8]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    let mut result: i64 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        result |= (b as i64) << (i * 8);
    }
    // Sign extend
    if bytes.last().unwrap() & 0x80 != 0 {
        let shift = bytes.len() * 8;
        if shift < 64 {
            result |= !0i64 << shift;
        }
    }
    result
}

fn pop_value(stack: &mut Vec<StackItem>) -> Result<PickleValue> {
    match stack.pop() {
        Some(StackItem::Value(v)) => Ok(v),
        Some(StackItem::Mark) => Err(Error::from_reason(
            "pickle: unexpected MARK when expecting value".to_string(),
        )),
        None => Err(Error::from_reason("pickle: stack underflow".to_string())),
    }
}

fn pop_mark(stack: &mut Vec<StackItem>) -> Result<Vec<PickleValue>> {
    let mut items = Vec::new();
    loop {
        match stack.pop() {
            Some(StackItem::Mark) => {
                items.reverse();
                return Ok(items);
            }
            Some(StackItem::Value(v)) => items.push(v),
            None => return Err(Error::from_reason("pickle: MARK not found".to_string())),
        }
    }
}

fn items_to_pairs(items: Vec<PickleValue>) -> Result<Vec<(PickleValue, PickleValue)>> {
    if !items.len().is_multiple_of(2) {
        return Err(Error::from_reason(
            "pickle: odd number of items for dict".to_string(),
        ));
    }
    let mut pairs = Vec::new();
    let mut iter = items.into_iter();
    while let Some(key) = iter.next() {
        let val = iter.next().unwrap();
        pairs.push((key, val));
    }
    Ok(pairs)
}

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const intTypes = [_]type{ i8, i16, i32, i64, u8, u16, u32, u64 };
const floatTypes = [_]type{ f32, f64 };

const TensorError = error{
    InvalidShape,
    WrongTypes,
    InvalidIndex,
    Dummy,
};

//{null, null, null} -> [:]
//{x, null, null} -> [x:]
//{null, x, null} -> [:x]
//{x,y,null} -> [x:y]
//{x,y,z} -> [x:y:z]
//{null, null, x} -> [::z]
pub const TensorRange = [3]?i32;
inline fn getTensorRangeLen(range: *const TensorRange, default_len: i32) i32 {
    const step: i32 = range[2] orelse 1;
    if (std.math.divCeil(i32, (range[1] orelse default_len) - (range[0] orelse 0), step)) |res| {
        return res;
    } else |_| {
        return 0;
    }
}
pub const FullRange = TensorRange{ null, null, null };

pub const TensorIdx = union(enum) {
    idx: i32,
    range: TensorRange,
    //TODO add slice

    inline fn getLen(self: TensorIdx, len: i32) i32 {
        switch (self) {
            .idx => return 1,
            .range => |range| return getTensorRangeLen(&range, len),
        }
    }
};

fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();
        data: []T,
        shape: []i32,
        rank: i32,
        // size: i32,
        alloc: std.mem.Allocator,
        offsets: []usize,
        grad: ?*Self,

        fn init(alloc: Allocator, shape: []const i32) !Tensor(T) {
            switch (@typeInfo(T)) {
                .Int => {},
                .Float => {},
                .Bool => {},
                else => {
                    std.debug.print("{s} is not supported\n", .{@typeName(T)});
                    return TensorError.InvalidShape;
                },
            }

            if (shapeIsValidForTensor(shape) == false) {
                const shape_string = try sliceToString(i32, alloc, shape);
                defer alloc.free(shape_string);
                std.debug.print("Tensor shape {s}", .{shape_string});
                return TensorError.InvalidShape;
            }
            var data_size: i32 = 1;
            for (shape) |el| {
                data_size *= el;
            }
            const offsets = try alloc.alloc(usize, shape.len);
            for (0..shape.len) |i| {
                const idx = shape.len - 1 - i;
                switch (i) {
                    0 => offsets[idx] = 0,
                    1 => offsets[idx] = @intCast(shape[idx + 1]),
                    else => offsets[idx] = offsets[idx + 1] * @as(usize, @intCast(shape[idx + 1])),
                }
            }
            return Tensor(T){ .data = try alloc.alloc(T, @intCast(data_size)), .shape = try alloc.dupe(i32, shape), .rank = @intCast(shape.len), .alloc = alloc, .grad = null, .offsets = offsets };
        }

        pub fn clone(self: *const Self, alloc: std.mem.Allocator) !Tensor(T) {
            const tensor = try Tensor(T).init(alloc, self.shape);
            errdefer tensor.deinit();
            @memcpy(tensor.data, self.data);
            return tensor;
        }

        pub fn dtype(self: *const Self, comptime DT: type, alloc: std.mem.Allocator) !Tensor(DT) {
            const new_tensor = try Tensor(DT).init(alloc, self.shape);
            errdefer new_tensor.deinit();
            const target = DT;
            const source = T;

            for (self.data, 0..) |el, i| {
                switch (@typeInfo(target)) {
                    .Int => |t| {
                        switch (@typeInfo(source)) {
                            .Int => |s| {
                                if (t.bits > s.bits) {
                                    new_tensor.data[i] = @as(target, el);
                                } else {
                                    new_tensor.data[i] = @as(target, @intCast(el));
                                }
                            },
                            .Float => new_tensor.data[i] = @as(target, @intFromFloat(el)),
                            else => {},
                        }
                    },
                    .Float => |t| {
                        switch (@typeInfo(source)) {
                            .Int => new_tensor.data[i] = @as(target, @floatFromInt(el)),
                            .Float => |s| {
                                if (t.bits > s.bits) {
                                    new_tensor.data[i] = @as(target, el);
                                } else {
                                    new_tensor.data[i] = @as(target, @floatCast(el));
                                }
                            },
                            else => {},
                        }
                    },
                    else => {
                        std.debug.print("{s} is not supported\n", .{@typeName(target)});
                        return TensorError.InvalidShape;
                    },
                }
            }
            return new_tensor;
        }

        //How to define an indexing operator?
        //- anon structs / tuples are required to be comp time known
        //- index {1, -1, 2} and separate range {":", ".", ":"} parameter => harder to use
        //- slice of unions {.{.i=0}, .{i=1}, .{range=[3]int{}}}
        pub fn get(self: *const Self, idxs: []const TensorIdx, alloc: std.mem.Allocator) !Tensor(T) {
            if (try tensorIdxIsValid(self.alloc, self.shape, idxs) == false) return TensorError.InvalidIndex;

            //compute new shape of tensor
            const new_shape: []i32 = try alloc.dupe(i32, self.shape);
            defer alloc.free(new_shape);
            for (idxs, 0..) |idx, i| {
                new_shape[i] = idx.getLen(self.shape[i]);
            }

            var new_tensor = try Tensor(T).init(alloc, new_shape);
            errdefer new_tensor.deinit();

            //expand idxs
            const exp_idxs = try alloc.alloc(TensorIdx, new_tensor.shape.len);
            defer alloc.free(exp_idxs);
            @memcpy(exp_idxs[0..idxs.len], idxs);
            if (exp_idxs.len > idxs.len) @memset(exp_idxs[idxs.len..], TensorIdx{ .range = .{ null, null, null } });

            var data_idxs = try std.ArrayList(usize).initCapacity(alloc, new_tensor.shape.len);
            defer data_idxs.deinit();
            const copied = try self.sliceRec(&new_tensor, exp_idxs, 0, &data_idxs, 0);
            std.debug.assert(copied > 0);

            //TODO deduce which dimensions to remove

            return new_tensor;
        }

        fn sliceRec(self: *const Self, to: *Self, idxs: []const TensorIdx, idx_pos: usize, data_idxs: *std.ArrayList(usize), to_offset: usize) !usize {
            //idx_pos describes the position sliced idxs: idxs[idx_pos:]
            //if(idxs describes successive memory)
            {
                const s = try tensorIdxToString(to.alloc, idxs, null);
                defer to.alloc.free(s);
                const di = try sliceToString(usize, to.alloc, data_idxs.items);
                defer to.alloc.free(di);
                std.debug.print("{s} - data idxs: {s}\n", .{ s, di });
            }

            std.debug.assert(idxs.len == self.shape.len); //idxs must be expanded
            std.debug.assert(data_idxs.items.len <= idxs.len);
            //TODO assert all elements are ranges of lists
            const new_idxs = idxs[idx_pos..];
            var ok = true;
            for (new_idxs, 0..) |idx, i| {
                switch (idx) {
                    .idx => |id| {
                        if (!(self.shape[i] == 1 and id == 0)) ok = false;
                    },
                    .range => |r| {
                        if (!(r[0] == null and r[1] == null and r[2] == null)) ok = false;
                    },
                }
                if (ok == false) break;
            }
            if (ok or new_idxs.len == 1) {
                std.debug.print("Reached recursive break cond\n", .{});
                defer std.debug.print("Recursive break cond end\n", .{});
                if (idx_pos == 0) { //this only means we slice the complete tensor
                    // std.debug.print("{d} {d}\n", .{ self.data.len, to.data.len });
                    std.debug.assert(self.data.len == to.data.len);
                    @memcpy(to.data, self.data);
                    return self.data.len;
                } else {
                    const ss = try sliceToString(i32, to.alloc, to.shape);
                    defer to.alloc.free(ss);
                    std.debug.print("to shape {s} - di len {d}\n", .{ ss, data_idxs.items.len });

                    const old_len = data_idxs.items.len;
                    for (0..to.shape.len - data_idxs.items.len) |i| {
                        switch (new_idxs[i]) {
                            .idx => |idx| try data_idxs.append(@intCast(idx)),
                            .range => try data_idxs.append(0),
                        }
                    }

                    const start = getIndex(self.shape, self.offsets, data_idxs.items);

                    var length: usize = 1;
                    for (to.shape[idx_pos..]) |d| length *= @intCast(d);

                    std.debug.print("{d} {d} {d} {d} {?}\n", .{ to_offset, start, length, idx_pos, ok });
                    const s = try tensorIdxToString(to.alloc, idxs, null);
                    defer to.alloc.free(s);
                    const di = try sliceToString(usize, to.alloc, data_idxs.items);
                    defer to.alloc.free(di);
                    std.debug.print("{s} - data idxs: {s}\n", .{ s, di });

                    @memcpy(to.data[to_offset .. to_offset + length], self.data[start .. start + length]);
                    try data_idxs.resize(old_len);
                    return length;
                }
            }

            var total_copied: usize = 0;
            try data_idxs.append(undefined);
            switch (new_idxs[0]) {
                .idx => |idx| {
                    data_idxs.items[data_idxs.items.len - 1] = @intCast(idx);
                    total_copied += try sliceRec(self, to, idxs, idx_pos + 1, data_idxs, to_offset + total_copied);
                },
                .range => |range| {
                    const start = range[0] orelse 0;
                    const end = range[1] orelse self.shape[idx_pos];
                    const step = range[2] orelse 1;
                    var i: usize = @intCast(start);
                    while (i < end) : (i += @intCast(step)) {
                        data_idxs.items[data_idxs.items.len - 1] = @intCast(i);
                        total_copied += try sliceRec(self, to, idxs, idx_pos + 1, data_idxs, to_offset + total_copied);
                    }
                },
            }
            try data_idxs.resize(data_idxs.items.len - 1);
            return total_copied;
        }

        // pub fn reshape(self: *const Self, shape: []i32) !Tensor {
        //     return TensorError.Dummy;
        // }
        // pub fn transpose(self: *const Self, shape: []i32) !Tensor {
        //     return TensorError.Dummy;
        // }

        //mainly for testing stuff
        pub fn dummy(self: *Self) void {
            dummyOp(T, self) catch |e| {
                if (@errorReturnTrace()) |s_trace| {
                    std.builtin.panicUnwrapError(s_trace, e);
                }
                // unreachable;
            };
        }

        pub fn deinit(self: *const Self) void {
            self.alloc.free(self.data);
            self.alloc.free(self.shape);
            self.alloc.free(self.offsets);
            if (self.grad) |grad| self.alloc.destroy(grad);
        }
    };
}

pub fn zeros(comptime T: type, alloc: std.mem.Allocator, shape: []const i32) !Tensor(T) {
    const tensor = try Tensor(T).init(alloc, shape);
    @memset(tensor.data, 0);
    return tensor;
}

pub fn ones(comptime T: type, alloc: std.mem.Allocator, shape: []const i32) !Tensor(T) {
    const tensor = try Tensor(T).init(alloc, shape);
    @memset(tensor.data, 1);
    return tensor;
}

pub fn fromSlice(comptime T: type, alloc: std.mem.Allocator, data: []const T, shape: []const i32) !Tensor(T) {
    var tensor = try Tensor(T).init(alloc, shape);
    if (getSizeFromShape(shape) != data.len) return TensorError.InvalidShape;
    tensor.alloc.free(tensor.data);
    tensor.data = try tensor.alloc.dupe(T, data);
    return tensor;
}

//TODO finish; requires float range to function properly --> needs rework of tensor range
pub fn fromRange(comptime T: type, alloc: Allocator, range: TensorRange, shape: []const i32) !Tensor(T) {
    var tensor = try Tensor(T).init(alloc, shape);
    errdefer tensor.deinit();
    if (getSizeFromShape(shape) != getTensorRangeLen(&range, tensor.data.len)) {
        const shape_string = sliceToString(i32, shape, alloc);
        defer alloc.free(shape_string);

        //TODO print range
        std.debug.print("Shape {s} does not fit to range", .{shape_string});
        return TensorError.InvalidShape;
    }
    const start = range[0] orelse 0;
    const end = range[1] orelse tensor.data.len;
    const step = range[2] orelse 1;
    var i = start;
    var j: usize = 0;
    while (i < end) : (i += step) {
        defer j += 1;
    }
    return TensorError.Dummy();
}

fn dummyOp(comptime T: type, t: *const Tensor(T)) !void {
    _ = t;
    return TensorError.Dummy;
}

//helpers -------------------------------------------------------------------------------------------------------------
inline fn shapeIsValidForTensor(shape: []const i32) bool {
    if (shape.len == 0) return false;
    for (shape) |el| {
        if (el <= 0) {
            return false;
        }
    }
    return true;
}

inline fn getSizeFromShape(shape: []const i32) i32 {
    if (shape.len == 0) return 0;
    var size: i32 = 1;
    for (shape) |el| {
        size *= el;
    }
    return size;
}
//caller must free string
fn tensorIdxToString(alloc: Allocator, idxs: []const TensorIdx, markAt: ?usize) ![]u8 {
    const substrings = try alloc.alloc([]u8, idxs.len);

    var i: usize = 0;
    defer {
        //free exact numbers of allocations
        for (0..i) |j| alloc.free(substrings[j]);
        alloc.free(substrings);
    }
    while (i < idxs.len) : (i += 1) {
        const field = idxs[i];
        switch (field) {
            .idx => |idx| {
                if (markAt) |mark| {
                    if (mark == i) {
                        substrings[i] = try std.fmt.allocPrint(alloc, "~~{d}~~", .{idx});
                        continue;
                    }
                }
                substrings[i] = try std.fmt.allocPrint(alloc, "{d}", .{idx});
            },
            .range => |range| {
                if (markAt) |mark| {
                    if (mark == i) {
                        substrings[i] = try std.fmt.allocPrint(alloc, "~~[{?}, {?}, {?}]~~", .{ range[0], range[1], range[2] });
                        continue;
                    }
                }
                substrings[i] = try std.fmt.allocPrint(alloc, "[{?}, {?}, {?}]", .{ range[0], range[1], range[2] });
            },
        }
    }

    const idx_string = try std.mem.join(alloc, ", ", substrings);
    defer alloc.free(idx_string);

    return std.fmt.allocPrint(alloc, "Tensor Index: [{s}]", .{idx_string});
}

fn sliceToString(comptime T: type, alloc: Allocator, data: []const T) ![]u8 {
    var substrings = try std.ArrayList([]u8).initCapacity(alloc, data.len);
    defer substrings.deinit();
    if (@typeInfo(T) != .Int and @typeInfo(T) != .Float) {
        return TensorError.WrongTypes;
    }
    for (data) |d| {
        try substrings.append(try std.fmt.allocPrint(alloc, "{d}", .{d}));
    }
    defer {
        for (substrings.items) |ss| {
            alloc.free(ss);
        }
    }
    const full_string = try std.mem.join(alloc, ", ", substrings.items);
    return full_string;
}

fn tensorIdxIsValid(alloc: Allocator, shape: []const i32, idxs: []const TensorIdx) !bool {
    std.debug.assert(shape.len > 0);
    //check if the overall length fits
    if (shape.len < idxs.len) {
        std.debug.print("Expects shape.len <= idx.len, but shape.len={d} and idx.len={d}\n", .{ shape.len, idxs.len });
        return false;
    }
    if (idxs.len == 0) {
        std.debug.print("Expects idx.len > 0, but idx.len = 0\n", .{});
        return false;
    }

    //check if there are bounds violations
    for (0..idxs.len) |i| {
        const field = idxs[i];
        switch (field) {
            .idx => |idx| {
                // std.debug.print("{d}\n", .{idx});
                //index in bounds
                if (idx > shape[i]) {
                    const idx_str = try tensorIdxToString(alloc, idxs, i);
                    defer alloc.free(idx_str);
                    std.debug.print("Invalid index in {s} for shape [", .{idx_str});
                    for (shape) |s| {
                        std.debug.print(" {d}", .{s});
                    }
                    std.debug.print(" ]\n", .{});
                    return false;
                }
            },
            .range => |range| {
                //TODO Fix me range violation not correctly detected: Check if all indices are in shape !
                const s = shape[i];
                const start = range[0] orelse 0;
                const end = range[1] orelse s;
                const step = range[2] orelse 1;
                const start_wrong = start < 0 or start >= s;
                const end_wrong = end < 0 or (getTensorRangeLen(&range, s) - 1) * step + start >= s;
                if (start_wrong or end_wrong) {
                    const idx_str = try tensorIdxToString(alloc, idxs, i);
                    defer alloc.free(idx_str);
                    std.debug.print("Invalid range in {s} for shape [", .{idx_str});
                    for (shape) |sh| {
                        std.debug.print(" {d}", .{sh});
                    }
                    std.debug.print(" ]\n", .{});
                    return false;
                }
            },
        }
    }

    return true;
}

//computes the index in .data
fn getIndex(shape: []const i32, offsets: []const usize, idxs: []const usize) usize {
    std.debug.assert(shape.len > 0);
    std.debug.assert(offsets.len == shape.len);
    std.debug.assert(offsets[offsets.len - 1] == 0);
    std.debug.assert(idxs.len == offsets.len);
    var sum: usize = idxs[idxs.len - 1];
    for (0..offsets.len - 1) |i| {
        sum += offsets[i] * idxs[i];
    }
    return sum;
}

//Tests -------------------------------------------------------------------------------------------------------------
test "zero tensor" {
    const alloc = testing.allocator;
    var tensor = try zeros(f32, alloc, &[_]i32{ 2, 3 });
    try testing.expectEqual(2, tensor.rank);
    try testing.expectEqual(6, tensor.data.len);
    defer tensor.deinit();
    const expected_data = [_]f32{0} ** 6;
    try testing.expectEqualSlices(f32, &expected_data, tensor.data);
}

test "ones tensor" {
    const alloc = testing.allocator;
    var tensor = try ones(f32, alloc, &[_]i32{ 2, 3 });
    try testing.expectEqual(2, tensor.rank);
    try testing.expectEqual(6, tensor.data.len);
    defer tensor.deinit();
    const expected_data = [_]f32{1} ** 6;
    try testing.expectEqualSlices(f32, &expected_data, tensor.data);
}

test "from slice tensor" {
    const alloc = testing.allocator;
    var tensor = try fromSlice(f32, alloc, &[_]f32{ 1, 2, 3, 4, 5, 6 }, &[_]i32{ 2, 3 });
    try testing.expectEqual(2, tensor.rank);
    try testing.expectEqual(6, tensor.data.len);
    defer tensor.deinit();
    const expected_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try testing.expectEqualSlices(f32, &expected_data, tensor.data);
}

test "clone tensor" {
    const alloc = testing.allocator;
    var tensor = try fromSlice(f32, alloc, &[_]f32{ 1, 2, 3, 4, 5, 6 }, &[_]i32{ 2, 3 });
    defer tensor.deinit();

    var cloned = try tensor.clone(alloc);
    defer cloned.deinit();
    try testing.expectEqual(tensor.rank, cloned.rank);
    try testing.expectEqual(tensor.data.len, cloned.data.len);
    const expected_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    try testing.expectEqualSlices(f32, &expected_data, tensor.data);
}

test "change dtye of tensor - f32 to f64" {
    const alloc = testing.allocator;
    // var tensor = try fromSlice(f32, alloc, &[_]f32{ 1, 2, 3, 4, 5, 6 }, &[_]i32{ 2, 3 });
    var tensor = try zeros(f32, alloc, &[_]i32{ 2, 3 });
    defer tensor.deinit();

    const changed_tensor = try tensor.dtype(f64, alloc);
    defer changed_tensor.deinit();

    try testing.expectEqualSlices(i32, tensor.shape, changed_tensor.shape);
}

test "change dtye of tensor - f64 to f32" {
    const alloc = testing.allocator;
    // var tensor = try fromSlice(f32, alloc, &[_]f32{ 1, 2, 3, 4, 5, 6 }, &[_]i32{ 2, 3 });
    var tensor = try zeros(f64, alloc, &[_]i32{ 2, 3 });
    defer tensor.deinit();

    const changed_tensor = try tensor.dtype(f32, alloc);
    defer changed_tensor.deinit();

    try testing.expectEqualSlices(i32, tensor.shape, changed_tensor.shape);
}

test "change dtye of tensor - i64 to i32" {
    const alloc = testing.allocator;
    // var tensor = try fromSlice(f32, alloc, &[_]f32{ 1, 2, 3, 4, 5, 6 }, &[_]i32{ 2, 3 });
    var tensor = try zeros(i64, alloc, &[_]i32{ 2, 3 });
    defer tensor.deinit();

    const changed_tensor = try tensor.dtype(i32, alloc);
    defer changed_tensor.deinit();
}

test "change dtye of tensor - i32 to i64" {
    const alloc = testing.allocator;
    // var tensor = try fromSlice(f32, alloc, &[_]f32{ 1, 2, 3, 4, 5, 6 }, &[_]i32{ 2, 3 });
    var tensor = try zeros(i32, alloc, &[_]i32{ 2, 3 });
    defer tensor.deinit();

    const changed_tensor = try tensor.dtype(i64, alloc);
    defer changed_tensor.deinit();
}

test "change dtye of tensor - f32 to i64" {
    const alloc = testing.allocator;
    // var tensor = try fromSlice(f32, alloc, &[_]f32{ 1, 2, 3, 4, 5, 6 }, &[_]i32{ 2, 3 });
    var tensor = try zeros(f32, alloc, &[_]i32{ 2, 3 });
    defer tensor.deinit();

    const changed_tensor = try tensor.dtype(i64, alloc);
    defer changed_tensor.deinit();
}

test "change dtye of tensor - i64 to f32" {
    const alloc = testing.allocator;
    // var tensor = try fromSlice(f32, alloc, &[_]f32{ 1, 2, 3, 4, 5, 6 }, &[_]i32{ 2, 3 });
    var tensor = try zeros(i64, alloc, &[_]i32{ 2, 3 });
    defer tensor.deinit();

    const changed_tensor = try tensor.dtype(f32, alloc);
    defer changed_tensor.deinit();
}

test "get sub tensor wrong indices" {
    const alloc = testing.allocator;
    // var tensor = try fromSlice(f32, alloc, &[_]f32{ 1, 2, 3, 4, 5, 6 }, &[_]i32{ 2, 3 });
    var tensor = try zeros(i64, alloc, &[_]i32{ 2, 3 });
    defer tensor.deinit();

    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]TensorIdx{ .{ .idx = 4 }, .{ .range = .{ 1, null, null } } }, alloc));
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]TensorIdx{ .{ .idx = 1 }, .{ .range = .{ 3, null, null } } }, alloc));
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]TensorIdx{ .{ .idx = 1 }, .{ .range = .{ null, 4, null } } }, alloc));
}

test "getIndex" {
    const alloc = testing.allocator;
    // var tensor = try fromSlice(f32, alloc, &[_]f32{ 1, 2, 3, 4, 5, 6 }, &[_]i32{ 2, 3 });
    var tensor = try zeros(i64, alloc, &[_]i32{ 4, 3, 2 });
    defer tensor.deinit();

    const idxs = [_]usize{ 1, 2, 1 };
    try testing.expectEqualSlices(usize, &[_]usize{ 6, 2, 0 }, tensor.offsets);
    try testing.expectEqual(11, getIndex(tensor.shape, tensor.offsets, &idxs));
}

test "get" {
    //TODO This will be obsoloete very fast
    const alloc = testing.allocator;

    var tensor = try fromSlice(f32, alloc, &[_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }, &[_]i32{ 4, 3, 2 });
    defer tensor.deinit();
    {
        const a = try tensor.get(&[_]TensorIdx{.{ .idx = 0 }}, alloc);
        defer a.deinit();

        try testing.expectEqual(tensor.shape.len, a.shape.len);
        try testing.expectEqualSlices(i32, &[_]i32{ 1, 3, 2 }, a.shape);
        try testing.expectEqualSlices(f32, &[_]f32{ 0, 1, 2, 3, 4, 5 }, a.data);
    }

    {
        const b = try tensor.get(&[_]TensorIdx{ .{ .idx = 0 }, .{ .idx = 1 } }, alloc);
        defer b.deinit();

        try testing.expectEqual(tensor.shape.len, b.shape.len);
        try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2 }, b.shape);
        try testing.expectEqualSlices(f32, &[_]f32{ 2, 3 }, b.data);
    }

    {
        const c = try tensor.get(&[_]TensorIdx{ .{ .idx = 0 }, .{ .idx = 1 }, .{ .idx = 1 } }, alloc);
        defer c.deinit();

        try testing.expectEqual(tensor.shape.len, c.shape.len);
        try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 1 }, c.shape);
        try testing.expectEqualSlices(f32, &[_]f32{3}, c.data);
    }

    {
        const d = try tensor.get(&[_]TensorIdx{
            .{ .range = FullRange },
            .{ .idx = 1 },
        }, alloc);
        defer d.deinit();

        try testing.expectEqual(tensor.shape.len, d.shape.len);
        try testing.expectEqualSlices(i32, &[_]i32{ 4, 1, 2 }, d.shape);
        try testing.expectEqualSlices(f32, &[_]f32{ 2, 3, 8, 9, 14, 15, 20, 21 }, d.data);
    }

    {
        std.debug.print("----------------------------\n", .{});
        const e = try tensor.get(&[_]TensorIdx{
            .{ .range = FullRange },
            .{ .range = TensorRange{ null, null, 2 } },
        }, alloc);
        defer e.deinit();

        try testing.expectEqual(tensor.shape.len, e.shape.len);
        try testing.expectEqualSlices(i32, &[_]i32{ 4, 2, 2 }, e.shape);
        try testing.expectEqualSlices(f32, &[_]f32{ 0, 1, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 18, 19, 22, 23 }, e.data);
    }

    {
        std.debug.print("----------------------------\n", .{});
        const f = try tensor.get(&[_]TensorIdx{
            .{ .range = FullRange },
            .{ .range = TensorRange{ 1, null, 2 } },
        }, alloc);
        defer f.deinit();

        try testing.expectEqual(tensor.shape.len, f.shape.len);
        try testing.expectEqualSlices(i32, &[_]i32{ 4, 1, 2 }, f.shape);
        try testing.expectEqualSlices(f32, &[_]f32{ 2, 3, 8, 9, 14, 15, 20, 21 }, f.data);
    }
    {
        std.debug.print("----------------------------\n", .{});
        const g = try tensor.get(&[_]TensorIdx{
            .{ .range = TensorRange{ 0, 3, 1 } },
            .{ .idx = 0 },
            .{ .range = TensorRange{ 0, 2, 1 } },
        }, alloc);
        defer g.deinit();

        try testing.expectEqual(tensor.shape.len, g.shape.len);
        try testing.expectEqualSlices(i32, &[_]i32{ 3, 1, 2 }, g.shape);
        try testing.expectEqualSlices(f32, &[_]f32{ 0, 1, 6, 7, 12, 13 }, g.data);
    }
    {
        std.debug.print("----------------------------\n", .{});
        const h = try tensor.get(&[_]TensorIdx{
            .{ .range = FullRange },
            .{ .range = FullRange },
            .{ .idx = 0 },
        }, alloc);
        defer h.deinit();

        try testing.expectEqual(tensor.shape.len, h.shape.len);
        try testing.expectEqualSlices(i32, &[_]i32{ 4, 3, 1 }, h.shape);
        try testing.expectEqualSlices(f32, &[_]f32{ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 }, h.data);
    }
}

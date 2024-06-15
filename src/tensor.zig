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
pub const FullRange = TensorRange{ null, null, null };

pub const TensorIdx = union(enum) {
    idx: i32,
    range: TensorRange,
};

fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();
        data: []T,
        shape: []i32,
        rank: i32,
        // size: i32,
        alloc: std.mem.Allocator,
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

            if (shapeIsValidForTensor(shape) == false) return TensorError.InvalidShape;
            var data_size: i32 = 1;
            for (shape) |el| {
                data_size *= el;
            }
            return Tensor(T){ .data = try alloc.alloc(T, @intCast(data_size)), .shape = try alloc.dupe(i32, shape), .rank = @intCast(shape.len), .alloc = alloc, .grad = null };
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
        pub fn get(self: *const Self, idxs: []const TensorIdx) !Tensor(T) {
            if (try tensorIdxIsValid(self.alloc, self.shape, idxs) == false) return TensorError.InvalidIndex;
            for (0..idxs.len) |i| {
                const d = idxs.len - 1 - i;
                const field = idxs[d];
                switch (field) {
                    .idx => |idx| {
                        // std.debug.print("{d}\n", .{idx});
                        _ = idx;
                    },
                    .range => |range| {
                        _ = range;
                        // for (range) |j| {
                        //     std.debug.print("{?} ", .{j});
                        // }
                        // std.debug.print("\n", .{});
                    },
                }
            }

            return TensorError.Dummy;
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
inline fn tensorIdxToString(alloc: Allocator, idxs: []const TensorIdx, markAt: ?usize) ![]u8 {
    const substrings = try alloc.alloc([]u8, idxs.len);
    defer {
        for (substrings) |ss| alloc.free(ss);
        alloc.free(substrings);
    }
    for (0..idxs.len) |i| {
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

fn tensorIdxIsValid(alloc: Allocator, shape: []const i32, idxs: []const TensorIdx) !bool {
    std.debug.assert(shape.len > 0);
    //check if the overall length fits
    if (shape.len > idxs.len) {
        std.debug.print("Expects idx.len <= dim.len, but idxs.len={d} and dim.len{d}\n", .{ shape.len, idxs.len });
        return false;
    }
    if (idxs.len == 0) {
        std.debug.print("Expects idx.len > 0, but idx.len = 0\n", .{});
        return false;
    }

    //check if there are bounds violations
    for (0..idxs.len) |i| {
        const d = idxs.len - 1 - i;
        const field = idxs[d];
        switch (field) {
            .idx => |idx| {
                // std.debug.print("{d}\n", .{idx});
                //index in bounds
                if (idx > shape[shape.len - 1 - i]) {
                    const idx_str = try tensorIdxToString(alloc, idxs, d);
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
                const s = shape[shape.len - 1 - i];
                const start_wrong = range[0] != null and (range[0].? < 0 or range[0].? >= s);
                const end_wrong = range[1] != null and (range[1].? < 0 or range[1].? >= s);
                if (start_wrong or end_wrong) {
                    const idx_str = try tensorIdxToString(alloc, idxs, d);
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

    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]TensorIdx{ .{ .idx = 4 }, .{ .range = .{ 1, null, null } } }));
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]TensorIdx{ .{ .idx = 1 }, .{ .range = .{ 3, null, null } } }));
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]TensorIdx{ .{ .idx = 1 }, .{ .range = .{ null, 3, null } } }));
}

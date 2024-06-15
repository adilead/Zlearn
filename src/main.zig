const std = @import("std");
const tensor = @import("tensor.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok);
    const alloc = gpa.allocator();
    const a = try tensor.ones(f32, alloc, &[_]i32{ 2, 3 });
    defer a.deinit();
    const b = try a.dtype(f32, alloc);
    defer b.deinit();
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

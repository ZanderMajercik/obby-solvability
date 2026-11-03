local DebugHelpers = {
    SHOULD_PRINT = true
}

function DebugHelpers.print(self, ...)
	if self.SHOULD_PRINT then
		print(...)
	end
end

return DebugHelpers
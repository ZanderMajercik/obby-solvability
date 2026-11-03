local initialFrame = script.Parent.CFrame
local initialVelocity = script.Parent.Velocity
local boundsDistance = script.Parent:GetAttribute("boundsDistance")
local RunService = game:GetService("RunService")

-- Reset initial conditions if out of bounds.
RunService.Stepped:connect(function()
    if (script.Parent.CFrame.Position - initialFrame.Position).Magnitude > boundsDistance then
        script.Parent.CFrame = initialFrame
        script.Parent.Velocity = initialVelocity
    end
end)
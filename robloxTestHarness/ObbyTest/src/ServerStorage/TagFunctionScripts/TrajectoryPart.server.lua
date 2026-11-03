-- TrajectoryPart.server.lua
-- Updates trajectory part transparency based on Roblox time and part's position in folder

local highlightPerTimestep = 120
local highlightSpeed = 1

local RunService = game:GetService("RunService")
local part = script.Parent
local partColor = part.Color
local partTimestep = part:GetAttribute("timestep")

if not partTimestep then
    return
end

-- Listen to Heartbeat and update transparency each frame based on Roblox time
RunService.Heartbeat:Connect(function()
    -- Get current time-based index (cycles through 0-3 based on elapsed time)
    -- Each part fades based on how far behind the active timestep it is.
    local currentTimeIndex = math.floor(tick() * highlightPerTimestep * highlightSpeed) % highlightPerTimestep
    local partIndex = (partTimestep - 1) % highlightPerTimestep
    local trailDistance = (currentTimeIndex - partIndex + highlightPerTimestep) % highlightPerTimestep * 2
    local brightness = math.max(0.2, 1 - (trailDistance / (highlightPerTimestep - 1)) * 0.8)
    local fadedColor = Color3.new(1, 1, 1):Lerp(partColor, 1 - brightness)

    -- Update color: the active part keeps its original color and previous parts form a trailing fade.
    if trailDistance == 0 then
        part.Color = partColor
    else
        part.Color = fadedColor
    end
end)
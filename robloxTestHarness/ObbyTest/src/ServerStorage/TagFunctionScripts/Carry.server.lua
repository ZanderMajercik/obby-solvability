local RunService = game:GetService("RunService")

DELTA_TIME = 0
local conditionsMetBindable = Instance.new("BindableEvent")
RunService.Stepped:connect(function(_, deltaTime)
    DELTA_TIME = deltaTime
    local RUN_FLAG = true
    for name, value in pairs(script:GetAttributes()) do
        attributeValue = script.Parent.Parent.States[name].Value
        -- print(name, attributeValue)
        if attributeValue == false then
            RUN_FLAG = false
            break
        end
    end
    if RUN_FLAG then
        conditionsMetBindable:Fire()
    end
end)
local conditionsMet = conditionsMetBindable.Event

-- apply velocity to the part so that the character moves with it when on top
local lastPosition = script.Parent.Position
local lastRotation = script.Parent.CFrame.Rotation

RunService.Stepped:connect(function ()
    local currPosition = script.Parent.Position
    local currRotation = script.Parent.CFrame.Rotation
    local lastAxis, lastAngle = lastRotation:ToAxisAngle()
    local currAxis, currAngle = currRotation:ToAxisAngle()

    local deltaPosition = currPosition - lastPosition
    local velocity = deltaPosition / DELTA_TIME

    script.Parent.AssemblyLinearVelocity = velocity
    script.Parent.AssemblyAngularVelocity = (currAngle - lastAngle) * currAxis / DELTA_TIME
    lastPosition = currPosition
    lastRotation = currRotation
end)
local RunService = game:GetService("RunService")
local motionType = script.Parent:GetAttribute("rotateType")
local angular = script.Parent:GetAttribute("angular")
local axis = script.Parent:GetAttribute("axis")

local DELTA_TIME = 0.016667
local conditionsMetBindable = Instance.new("BindableEvent")
RunService.Stepped:connect(function(_, deltaTime)
    --DELTA_TIME = deltaTime
    local RUN_FLAG = true

    -- Simulation control
    local simulate = script.Parent:GetAttribute("Simulate")
    local timestep = script.Parent:GetAttribute("Timestep")
    if simulate then
        timestep += 1
        script.Parent:SetAttribute("Timestep", timestep)
        if timestep > 36000 then
            timestep = 0
        end
    end
    
    for name, value in pairs(script:GetAttributes()) do
        attributeValue = script.Parent.Parent.States[name].Value
        -- print(name, attributeValue)
        if attributeValue == false then
            RUN_FLAG = false
            break
        end
    end
    if RUN_FLAG then
        conditionsMetBindable:Fire(timestep)
    end
end)
local conditionsMet = conditionsMetBindable.Event


-- Multiple behaviors may depend on this target.
local proxyToolTarget = script.Parent:FindFirstChild("ProxyToolTarget")
-- if not proxyToolTarget then
--     -- Create the proxy part an parent to the current block.
--     proxyToolTarget = Instance.new("Part")
--     proxyToolTarget.Name = "ProxyToolTarget"
--     proxyToolTarget.Transparency = 0.6
--     proxyToolTarget.Anchored = true
--     proxyToolTarget.CanCollide = false
--     proxyToolTarget.CanTouch = false
--     proxyToolTarget.CastShadow = false
--     proxyToolTarget.Parent = script.Parent
--     proxyToolTarget.CFrame = script.Parent.CFrame
--     proxyToolTarget.Size = Vector3.new(2,2,2)
--     proxyToolTarget.Color = Color3.new(0,0,0)
-- end

--local currentRotationCFrame = CFrame.new()
--local lastTimeStep = 0
local function setRotation(timestep)
    local rotationCFrame = nil
    if axis == 'right' then
        rotationCFrame =  CFrame.Angles(timestep * DELTA_TIME * angular, 0, 0)
    elseif axis == 'up' then
        rotationCFrame = CFrame.Angles(0, timestep * DELTA_TIME * angular, 0)
    elseif axis == 'forward' then
        rotationCFrame = CFrame.Angles(0, 0, timestep * DELTA_TIME * angular)
    end

    script.Parent.CFrame = proxyToolTarget.CFrame * rotationCFrame
end

if motionType == "PersistentRotate" then
    conditionsMet:connect(function(timestep)
        setRotation(timestep)
    end)
elseif motionType == "TouchRotate" then
    script.Parent.Touched:connect(function(hit)
        conditionsMet:connect(rotateStep)
    end)
end

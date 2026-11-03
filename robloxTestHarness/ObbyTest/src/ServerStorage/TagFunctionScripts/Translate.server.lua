local HttpService = game:GetService("HttpService")
local TweenService = game:GetService("TweenService")
local RunService = game:GetService("RunService")

local DELTA_TIME = 0.016667
local conditionsMetBindable = Instance.new("BindableEvent")
RunService.Heartbeat:connect(function(_, deltaTime)
    --DELTA_TIME = deltaTime
    local RUN_FLAG = true

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
        local attributeValue = nil
        if script.Parent.Parent.States:FindFirstChild(name) then
            attributeValue = script.Parent.Parent.States[name].Value
        end
        -- print(name, attributeValue)
        if attributeValue ~= nil then
            RUN_FLAG = attributeValue
            break
        end
    end
    if RUN_FLAG then
        conditionsMetBindable:Fire(timestep)
    end
end)
local conditionsMet = conditionsMetBindable.Event

local easingStyle = Enum.EasingStyle[script.Parent:GetAttribute("easingStyle")]
local easingDirection = Enum.EasingDirection[script.Parent:GetAttribute("easingDirection")]
local startPosition = HttpService:JSONDecode(script.Parent:GetAttribute("startPosition"))
local endPosition = HttpService:JSONDecode(script.Parent:GetAttribute("endPosition"))

local startPosition = Vector3.new(
    startPosition[1],
    startPosition[2],
    startPosition[3]
)
local endPosition = Vector3.new(
    endPosition[1],
    endPosition[2],
    endPosition[3]
)

local motionType = script.Parent:GetAttribute("translateType")
local time = script.Parent:GetAttribute("time")
local reverses = script.Parent:GetAttribute("reverses")
local loop = script.Parent:GetAttribute("loop")



local tweenVal = Instance.new("Vector3Value")
tweenVal.Value = startPosition

local goal = {}
goal.Value = endPosition

local originalPosition = script.Parent.Position

-- Multiple behaviors may depend on this target.
local proxyToolTarget = script.Parent:FindFirstChild("ProxyToolTarget")

-- Animate the translation offset
local tweenInfo = TweenInfo.new(time, easingStyle, easingDirection, loop, reverses)
local tween = TweenService:Create(tweenVal, tweenInfo, goal)

local forwardVelocity = (endPosition - startPosition) / time
script:SetAttribute("Velocity", forwardVelocity)
local currentPosition = startPosition

local touchConnection = nil
local lastTimeStep = -1
local numLoops = 0
if motionType == "PersistentTranslate" then
    conditionsMet:Connect(function(timestep)
        --tween:Play()
        if lastTimeStep == -1 then
            lastTimeStep = timestep
        end
        local timeDelta = timestep - lastTimeStep
        lastTimeStep = timestep
        -- In case the user moves the proxy target with the sim paused.
        script.Parent.CFrame = proxyToolTarget.CFrame * CFrame.new(currentPosition)
        -- Lua is a fake language, so we have to special case this outside the for loop.
        if timeDelta == 0 then
            script:SetAttribute("Velocity", forwardVelocity)
            return
        end
        for i = math.sign(timeDelta), timeDelta, math.sign(timeDelta) do
            -- Looping comparison.
            -- If time moves forward, max of "loop" loops. If backward, 0 loops.
            local timeCorrectedMaxLoops = math.max(math.sign(timeDelta) * loop, 0)
            local timeCorrectedNumLoops = math.sign(timeDelta) * numLoops
            if (timeCorrectedNumLoops <= timeCorrectedMaxLoops) then
                if (endPosition - currentPosition).Magnitude > (endPosition - startPosition).Magnitude then
                    forwardVelocity = math.sign(timeDelta) * (endPosition - startPosition) / time
                    if loop ~= -1 then
                        --Only track loops if not looping infinitely.
                        numLoops += math.sign(timeDelta)
                    end
                end
                if (startPosition - currentPosition).Magnitude > (endPosition - startPosition).Magnitude then
                    forwardVelocity = math.sign(timeDelta) * -(endPosition - startPosition) / time
                    if loop ~= -1 then
                        --Only track loops if not looping infinitely.
                        numLoops += math.sign(timeDelta)
                    end
                end
                currentPosition += forwardVelocity * DELTA_TIME * math.sign(timeDelta)
                script.Parent.CFrame = proxyToolTarget.CFrame * CFrame.new(currentPosition)
            end
        end
        script:SetAttribute("Velocity", forwardVelocity)
    end)
elseif motionType == "TouchTranslate" then
    conditionsMet:Connect(function()
        touchConnection = script.Parent.Touched:connect(function(hit)
            tween:Play()
            script.Parent.CFrame = proxyToolTarget.CFrame * CFrame.new(tweenVal.Value)
            touchConnection:Disconnect()
        end)
    end)
end
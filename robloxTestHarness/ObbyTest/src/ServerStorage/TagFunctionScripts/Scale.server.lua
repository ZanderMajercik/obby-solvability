local HttpService = game:GetService("HttpService")
local TweenService = game:GetService("TweenService")
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

local alignPosition = Instance.new("AlignPosition", script.Parent)
local alignOrientation = Instance.new("AlignOrientation", script.Parent)

local easingStyle = Enum.EasingStyle[script.Parent:GetAttribute("easingStyle")]
local easingDirection = Enum.EasingDirection[script.Parent:GetAttribute("easingDirection")]
local startScale = HttpService:JSONDecode(script.Parent:GetAttribute("startScale"))
local endScale = HttpService:JSONDecode(script.Parent:GetAttribute("endScale"))
script.Parent.Size = Vector3.new(
    startScale[1],
    startScale[2],
    startScale[3]
) * script.Parent.Size
endScale = Vector3.new(endScale[1], endScale[2], endScale[3])

local goal = {}
goal.Size = endScale

-- animate the part
local motionType = script.Parent:GetAttribute("scaleType")
local time = script.Parent:GetAttribute("time")
local reverses = script.Parent:GetAttribute("reverses")
local loop = script.Parent:GetAttribute("loop")
local tweenInfo = TweenInfo.new(time, easingStyle, easingDirection, loop, reverses)
local tween = TweenService:Create(script.Parent, tweenInfo, goal)
if motionType == "PersistentScale" then
    conditionsMet:Connect(function()
        tween:Play()
    end)
elseif motionType == "TouchScale" then
    conditionsMet:Connect(function()
        script.Parent.Touched:connect(function(hit)
            tween:Play()
        end)
    end)
end
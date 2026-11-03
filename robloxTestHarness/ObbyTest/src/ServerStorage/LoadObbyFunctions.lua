local LoadObbyFunctions = {}

local HttpService = game:GetService("HttpService")
local CollectionService = game:GetService("CollectionService")
local TagFunctions = require(game:GetService("ServerStorage").TagFunctions)
local TrainingBehaviorFunctions = require(game:GetService("ServerStorage").TrainingBehaviorFunctions)


-- See: https://create.roblox.com/docs/reference/engine/datatypes/CFrame#fromMatrix
local convertToZUpFrame = CFrame.fromMatrix(
    Vector3.new(0, 0, 0),
    Vector3.new(1, 0, 0), -- +Right Vector
    Vector3.new(0, 0, 1), -- +Up Vector
    -Vector3.new(0, 1, 0) -- -Forward Vector
)

local function convertToYUp(v)
	return convertToZUpFrame:Inverse() * v
end

local function convertToZUp(v)
    return convertToZUpFrame * v
end


-- Mapping from behavior tag to the name of the child script on the part
local behaviorScriptNames = {
    Unanchored = "OutOfBoundsReset",
    Damage = "Damage",
    Checkpoint = "Checkpoint",
    Velocity = "Velocity",
    Conveyor = "Velocity",
    Trampoline = "Velocity",
    Disappear = "Disappear",
    TouchTranslate = "Translate",
    PersistentTranslate = "Translate",
    TouchRotate = "Rotate",
    PersistentRotate = "Rotate",
    TouchScale = "Scale",
    PersistentScale = "Scale",
    Friction = "Friction",
    Carry = "Carry",
    HoldState = "HoldState",
    PressState = "PressState",
}

local function getConditionsFromScript(part, behaviorName, obbyPartNames)
    local conditions = {}
    local scriptName = behaviorScriptNames[behaviorName]
    if scriptName then
        local behaviorScript = part:FindFirstChild(scriptName)
        if behaviorScript then
            for name, value in pairs(behaviorScript:GetAttributes()) do
                if obbyPartNames[name] then
                    table.insert(conditions, name)
                end
            end
        end
    end
    return conditions
end

local function isBasicPart(blockPartType)
    for index, value in ipairs({"Ball", "Block", "Cylinder", "Wedge"}) do
        if value == blockPartType then
            return true
        end
    end
    return false
end

local function expandAABB(part, aabb)
    local corners = {
        Vector3.new(1, 1, 1),
        Vector3.new(-1, 1, 1),
        Vector3.new(1, -1, 1),
        Vector3.new(-1, -1, 1),
        Vector3.new(1, 1, -1),
        Vector3.new(-1, 1, -1),
        Vector3.new(1, -1, -1),
        Vector3.new(-1, -1, -1)
    }
    for _, c in corners do
        local local_corner = convertToZUp(part.CFrame * (0.5 * part.Size * c))
        if not aabb.pMin then
            aabb.pMin = local_corner
        end
        if not aabb.pMax then
            aabb.pMax = local_corner
        end
        aabb.pMax = aabb.pMax:Max(local_corner)
        aabb.pMin = aabb.pMin:Min(local_corner)
    end
end

-- Logically, creating the level should be totally separate from training,
-- but enforcing this completely would involve referencing part names as 
-- a way of inferring order.
-- So, we during level creation we add tags that the training code can 
-- operate over. Functionality is added later (see TrainingBehaviorFunctions.lua).
local function createBlockPart(blockPartName, blockPartInfo, sectionTagIndex, initialAABBs, stateFolder)
    local part = nil
    for idx, behavior in blockPartInfo["behaviors"] do
        if behavior["name"] == "Procedural" then
            return part
        end
    end

    if isBasicPart(blockPartInfo["type"]) then
        part = Instance.new("Part")
        part.Shape = blockPartInfo["type"]
    else
        part = Instance.new(blockPartInfo["type"])
    end
    part.Name = blockPartName

    -- NOTE: The vertical axis in Roblox is the Y-axis.
    part.Position = Vector3.new(blockPartInfo["position"][1], blockPartInfo["position"][2], blockPartInfo["position"][3])
    part.Size = Vector3.new(blockPartInfo["size"][1], blockPartInfo["size"][2], blockPartInfo["size"][3])
    part.Orientation = Vector3.new(blockPartInfo["orientation"][1], blockPartInfo["orientation"][2], blockPartInfo["orientation"][3])

    part.Parent = workspace -- Put the part into the Workspace
    part.Anchored = true

    -- Darker gray
    part.Color = Color3.fromRGB(98, 97, 99)
    if blockPartInfo["type"] == "SpawnLocation" then
        part.Color = Color3.fromRGB(39, 70, 45)
    end
    CollectionService:AddTag(part, tostring(sectionTagIndex.Value))

    -- To match Madrona, we need to get the AABB of the level *before* 
    -- behaviors start to move blocks around.
    if initialAABBs then
        -- Create a new AABB for this section if one does not exist.
        if not initialAABBs[sectionTagIndex.Value] then
            initialAABBs[sectionTagIndex.Value] = {
                pMin = nil,
                pMax = nil
            }
        end
        expandAABB(part, initialAABBs[sectionTagIndex.Value])
    end

    local partStates = Instance.new("Folder", part)
    partStates.Name = "States"

    local state = blockPartInfo["state"]
    local State = Instance.new("BoolValue", stateFolder)
    State.Name = state["name"]
    State.Value = state["value"]
    local State = Instance.new("BoolValue", partStates)
    State.Name = state["name"]
    State.Value = state["value"]

    for idx, behavior in blockPartInfo["behaviors"] do
        if blockPartInfo["type"] == "SpawnLocation" then
            CollectionService:AddTag(part, "Spawn")
        end
        local behaviorName = behavior["name"]
        CollectionService:AddTag(part, behaviorName)
        if behaviorName == "Checkpoint" then
            --sectionTagIndex.Value += 1
            if initialAABBs then
                if not initialAABBs[sectionTagIndex.Value + 1] then
                    initialAABBs[sectionTagIndex.Value + 1] = {
                        pMin = nil,
                        pMax = nil
                    }
                end
                expandAABB(part, initialAABBs[sectionTagIndex.Value + 1])
            end
            -- Checkpoints are part of both observation spaces.
            CollectionService:AddTag(part, tostring(sectionTagIndex.Value + 1))

            -- This is safe now that checkpoints are guaranteed to appear in order.
            for i, g in CollectionService:GetTagged("Goal") do
                CollectionService:RemoveTag(g, "Goal")
            end
            CollectionService:AddTag(part, "Goal")
        end

        -- For this to be functional, the episode should live on the module.
        local dummyEpisode = {
            success = false,
            observations = {},
            actions = {}
        }

        if CollectionService:HasTag(part, "Goal") then
            part.Color = Color3.new(0, 0, 1)
            part.Material = Enum.Material.Neon
            --part.Transparency = 0.2
            TrainingBehaviorFunctions["KillAndReportEpisode"](part, dummyEpisode)
        else
            attachTagFunction(part, behaviorName, behavior, stateFolder)
        end
    end

    --Add a tag to identify this as an obby so we can remove it from the workspace
    CollectionService:AddTag(part, "obby")
    return part
end

function attachTagFunction(part, behaviorName, behavior)
    -- There are two steps to attaching the behavior script. First, we need to
    -- associate all the behavior settings as attributes to the part. Then, we
    -- duplicate the corresponding behavior script and parent it to the part.
    local script = nil
    if behaviorName == "Unanchored" then
        part.Anchored = false
        part.Color = Color3.new(1, 0.5, 0)
        part:SetAttribute("boundsDistance", 100) -- This should be set relative to level bounding box.
        script = game.ServerStorage.TagFunctionScripts.OutOfBoundsReset:Clone()
        for idx, condition in behavior["conditions"] do
            script:SetAttribute(tostring(condition), 1)
        end
        script.Parent = part
        local testPhysicalProperties = PhysicalProperties.new(part.Material)
        part.CustomPhysicalProperties = PhysicalProperties.new(0.1 / part.Size.Y, 0.001, testPhysicalProperties.Elasticity)
    elseif behaviorName == "Damage" then
        part:SetAttribute("damage", behavior["damage"])
        part.Color = Color3.new(1, 0, 0)
        -- Neon adds emissive with interferes with trajectory visualization.
        part.Material = Enum.Material.Plastic
        --part.Transparency = 0.2
        script = game.ServerStorage.TagFunctionScripts.Damage:Clone()
        for idx, condition in behavior["conditions"] do
            script:SetAttribute(tostring(condition), 1)
        end
        script.Parent = part
    elseif behaviorName == "Checkpoint" then
        part.Color = Color3.new(0, 0, 1)
        part.Material = Enum.Material.Neon
        --part.Transparency = 0.2
        script = game.ServerStorage.TagFunctionScripts.Checkpoint:Clone()
        for idx, condition in behavior["conditions"] do
            script:SetAttribute(tostring(condition), 1)
        end
        script.Parent = part
    elseif behaviorName == "Velocity" or behaviorName == "Conveyor" or behaviorName == "Trampoline" then
        part:SetAttribute("velocity", HttpService:JSONEncode(behavior["velocity"]))
        part.Color = Color3.new(0.5, 0, 0.5)
        script = game.ServerStorage.TagFunctionScripts.Velocity:Clone()
        for idx, condition in behavior["conditions"] do
            script:SetAttribute(tostring(condition), 1)
        end
        script.Parent = part
    elseif behaviorName == "Disappear" then
        part:SetAttribute("disappearDelay", behavior["disappear_delay"])
        part:SetAttribute("reappearDelay", behavior["reappear_delay"])
        script = game.ServerStorage.TagFunctionScripts.Disappear:Clone()
        for idx, condition in behavior["conditions"] do
            script:SetAttribute(tostring(condition), 1)
        end
        script.Parent = part
    elseif behaviorName == "TouchTranslate" or behaviorName == "PersistentTranslate" then
        part:SetAttribute("translateType", behaviorName)
        part:SetAttribute("time", behavior["time"])
        part:SetAttribute("reverses", behavior["reverses"])
        part:SetAttribute("easingStyle", behavior["easing_style"])
        part:SetAttribute("easingDirection", behavior["easing_direction"])
        part:SetAttribute("startPosition", HttpService:JSONEncode(behavior["start_position"]))
        part:SetAttribute("endPosition", HttpService:JSONEncode(behavior["end_position"]))
        part:SetAttribute("loop", behavior["loop"])
        -- Add simulation controls
        part:SetAttribute("Simulate", true)
        part:SetAttribute("Timestep", 0)
        local proxyToolTarget = part:FindFirstChild("ProxyToolTarget")
        if not proxyToolTarget then
            -- Create the proxy part an parent to the current block.
            proxyToolTarget = Instance.new("Part")
            proxyToolTarget.Name = "ProxyToolTarget"
            proxyToolTarget.Transparency = 0.6
            proxyToolTarget.Anchored = true
            proxyToolTarget.CanCollide = false
            proxyToolTarget.CanTouch = false
            proxyToolTarget.CastShadow = false
            proxyToolTarget.Parent = part
            local p = part.Position
            local o = part.Orientation
            if behavior["base_position"] then
                p = Vector3.new(behavior["base_position"][1], behavior["base_position"][2], behavior["base_position"][3])
                o = Vector3.new(behavior["base_orientation"][1], behavior["base_orientation"][2], behavior["base_orientation"][3])
            end
            proxyToolTarget.CFrame = CFrame.new(p) * CFrame.fromOrientation(math.rad(o.X), math.rad(o.Y), math.rad(o.Z))
            proxyToolTarget.Size = Vector3.new(2,2,2)
            proxyToolTarget.Color = Color3.new(0,0,0)
        end
        script = game.ServerStorage.TagFunctionScripts.Translate:Clone()
        for idx, condition in behavior["conditions"] do
            script:SetAttribute(tostring(condition), 1)
        end
        script.Parent = part
    elseif behaviorName == "TouchRotate" or behaviorName == "PersistentRotate" then
        part:SetAttribute("rotateType", behaviorName)
        part:SetAttribute("angular", behavior["angular"])
        part:SetAttribute("axis", behavior["axis"])
        part:SetAttribute("loop", behavior["loop"])
        -- Add simulation controls
        part:SetAttribute("Simulate", true)
        part:SetAttribute("Timestep", 0)
        local proxyToolTarget = part:FindFirstChild("ProxyToolTarget")
        if not proxyToolTarget then
            -- Create the proxy part an parent to the current block.
            proxyToolTarget = Instance.new("Part")
            proxyToolTarget.Name = "ProxyToolTarget"
            proxyToolTarget.Transparency = 0.6
            proxyToolTarget.Anchored = true
            proxyToolTarget.CanCollide = false
            proxyToolTarget.CanTouch = false
            proxyToolTarget.CastShadow = false
            proxyToolTarget.Parent = part
            local p = part.Position
            local o = part.Orientation
            if behavior["base_position"] then
                p = Vector3.new(behavior["base_position"][1], behavior["base_position"][2], behavior["base_position"][3])
                o = Vector3.new(behavior["base_orientation"][1], behavior["base_orientation"][2], behavior["base_orientation"][3])
            end
            proxyToolTarget.CFrame = CFrame.new(p) * CFrame.fromOrientation(math.rad(o.X), math.rad(o.Y), math.rad(o.Z))
            proxyToolTarget.Size = Vector3.new(2,2,2)
            proxyToolTarget.Color = Color3.new(0,0,0)
        end
        script = game.ServerStorage.TagFunctionScripts.Rotate:Clone()
        for idx, condition in behavior["conditions"] do
            script:SetAttribute(tostring(condition), 1)
        end
        script.Parent = part
    elseif behaviorName == "TouchScale" or behaviorName == "PersistentScale" then
        part:SetAttribute("scaleType", behaviorName)
        part:SetAttribute("time", behavior["time"])
        part:SetAttribute("reverses", behavior["reverses"])
        part:SetAttribute("easingStyle", behavior["easing_style"])
        part:SetAttribute("easingDirection", behavior["easing_direction"])
        part:SetAttribute("startScale", HttpService:JSONEncode(behavior["start_scale"]))
        part:SetAttribute("endScale", HttpService:JSONEncode(behavior["end_scale"]))
        part:SetAttribute("loop", behavior["loop"])
        -- Add simulation controls
        part:SetAttribute("Simulate", true)
        part:SetAttribute("Timestep", 0)
        script = game.ServerStorage.TagFunctionScripts.Scale:Clone()
        for idx, condition in behavior["conditions"] do
            script:SetAttribute(tostring(condition), 1)
        end
        script.Parent = part
    elseif behaviorName == "Friction" then
        part:SetAttribute("friction", behavior["friction"])
        part:SetAttribute("frictionWeight", behavior["weight"])
        part.Color = Color3.new(0, 0.75, 1)
        script = game.ServerStorage.TagFunctionScripts.Friction:Clone()
        for idx, condition in behavior["conditions"] do
            script:SetAttribute(tostring(condition), 1)
        end
        script.Parent = part
    elseif behaviorName == "Carry" then
        script = game.ServerStorage.TagFunctionScripts.Carry:Clone()
        for idx, condition in behavior["conditions"] do
            script:SetAttribute(tostring(condition), 1)
        end
        script.Parent = part
    elseif behaviorName == "HoldState" then
        part.Color = Color3.new(0.75, 0.75, 0.25)
        script = game.ServerStorage.TagFunctionScripts.HoldState:Clone()
        script.Parent = part
    elseif behaviorName == "PressState" then
        part.Color = Color3.new(1, 1, 0)
        script = game.ServerStorage.TagFunctionScripts.PressState:Clone()
        script.Parent = part
    elseif behaviorName == "Waypoint" then
        part.BrickColor = BrickColor.new("Bright yellow")
        part.Transparency = 0.5
        CollectionService:AddTag(part, "Waypoint")

        -- Add a label above the waypoint
        local billboardGui = Instance.new("BillboardGui")
        billboardGui.Name = "WaypointLabel"
        billboardGui.Adornee = part
        billboardGui.Size = UDim2.new(0, 80, 0, 20)
        billboardGui.StudsOffset = Vector3.new(0, part.Size.Y / 2 + 1, 0)
        billboardGui.AlwaysOnTop = true

        local textLabel = Instance.new("TextLabel")
        textLabel.Name = "Label"
        textLabel.Size = UDim2.new(1, 0, 1, 0)
        textLabel.BackgroundTransparency = 1
        textLabel.Text = tostring(behavior["index"])
        textLabel.TextColor3 = Color3.new(1, 1, 0)
        textLabel.TextScaled = true
        textLabel.Font = Enum.Font.GothamBold
        textLabel.Parent = billboardGui

        billboardGui.Parent = part

        part:SetAttribute("index", behavior["index"])
        part:SetAttribute("jumps", behavior["jumps"] or -1)
    elseif behaviorName == "Avoid" then
        part.BrickColor = BrickColor.new("Bright red")
        part.Transparency = 0.5
        CollectionService:AddTag(part, "Avoid")
    elseif behaviorName == "NoCollide" then
        part.CanCollide = false
        CollectionService:AddTag(part, "NoCollide")
    elseif behaviorName == "Procedural" then
        CollectionService:AddTag(part, "Procedural")
    end
end

local function adjustWaypointLabels()
    local waypoints = CollectionService:GetTagged("Waypoint")
    local totalWaypoints = #waypoints
    for _, waypoint in waypoints do
        local index = waypoint:GetAttribute("index") or 0
        local adjustedIndex = 1 + totalWaypoints - index
        local billboardGui = waypoint:FindFirstChild("WaypointLabel")
        if billboardGui then
            local textLabel = billboardGui:FindFirstChild("Label")
            if textLabel then
                textLabel.Text = tostring(adjustedIndex)
            end
        end
    end
end

local function createPartJson(p, idx, sectionTagIndex, debugPositionOffset, obbyPartNames)
    local partJson = {}
    partJson["type"] = "Block"
    if p:IsA("Part") then
        partJson["type"] = string.sub(tostring(p.Shape), string.len("Enum.PartType.") + 1, string.len(tostring(p.Shape)))
    end
    partJson["class"] = "Block"

    local pos = p.Position - debugPositionOffset
    local orientation = p.Orientation

    -- Behaviors that induce a varying motion have a proxy part that stores the
    -- initial CFrame for purposes of using roblox tools to manipulate the part and
    -- computing the correct position/orientation when converting to JSON.
    -- local basePosition = pos
    -- local baseOrientation = orientation
    -- if p:FindFirstChild("ProxyToolTarget") then
    --     basePosition = p.ProxyToolTarget.Position - debugPositionOffset
    --     baseOrientation = p.ProxyToolTarget.Orientation
    -- end

    partJson["position"] = {pos.X, pos.Y, pos.Z}
    partJson["orientation"] = {orientation.X, orientation.Y, orientation.Z}
    partJson["size"] = {p.Size.X, p.Size.Y, p.Size.Z}

    if p:FindFirstChild("States") then
    -- Parts should only ever manage a single state.
        local state = p.States:GetChildren()[1]
        partJson["state"] = {
            name = state.Name,
            value = state.Value
        }
    else
        partJson["state"] = {
            name = "None",
            value = false
        }
    end


    partJson["behaviors"] = {}
    if CollectionService:HasTag(p, "Spawn") or CollectionService:HasTag(p, tostring(sectionTagIndex - 1)) then
        -- The spawn point for this section
        local behaviors = {}
        behaviors["name"] = "Spawn"
        behaviors["conditions"] = getConditionsFromScript(p, "Spawn", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
        partJson["type"] = "SpawnLocation"
    elseif CollectionService:HasTag(p, "Checkpoint") then
        -- The goal checkpoint for this section
        local behaviors = {}
        behaviors["name"] = "Checkpoint"
        behaviors["conditions"] = getConditionsFromScript(p, "Checkpoint", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    -- Truss behavior
    if CollectionService:HasTag(p, "Truss") then
        local behaviors = {}
        behaviors["name"] = "Truss"
        behaviors["conditions"] = getConditionsFromScript(p, "Truss", obbyPartNames)
        partJson["type"] = "TrussPart" -- Necessary for correct loading into Roblox.
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "Damage") then
        local behaviors = {}
        behaviors["name"] = "Damage"
        behaviors["damage"] = 100
        behaviors["conditions"] = getConditionsFromScript(p, "Damage", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    -- Process implicit rotation behavior. Y rotation only.
    if CollectionService:HasTag(p, "PersistentRotate") then
        local behaviors = {}
        behaviors["name"] = "PersistentRotate"
        for i, behaviorAttribute in pairs({"angular", "axis", "loop", "rotateType"}) do
            behaviors[behaviorAttribute] = p:GetAttribute(behaviorAttribute)
        end
        behaviors["base_position"] = {p.ProxyToolTarget.Position.X, p.ProxyToolTarget.Position.Y, p.ProxyToolTarget.Position.Z}
        behaviors["base_orientation"] = {p.ProxyToolTarget.Orientation.X, p.ProxyToolTarget.Orientation.Y, p.ProxyToolTarget.Orientation.Z}
        behaviors["conditions"] = getConditionsFromScript(p, "PersistentRotate", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "Carry") then
        local behaviors = {}
        behaviors["name"] = "Carry"
        behaviors["conditions"] = getConditionsFromScript(p, "Carry", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "PersistentTranslate") then
        local behaviors = {}
        behaviors["name"] = "PersistentTranslate"
    
        for i, behaviorAttribute in pairs({"loop", "reverses", "time", "translateType"}) do
            behaviors[behaviorAttribute] = p:GetAttribute(behaviorAttribute)
        end
        behaviors["easing_style"] = p:GetAttribute("easingStyle")
        behaviors["easing_direction"] = p:GetAttribute("easingDirection")

        behaviors["start_position"] = HttpService:JSONDecode(p:GetAttribute("startPosition"))
        behaviors["end_position"] = HttpService:JSONDecode(p:GetAttribute("endPosition"))

        behaviors["base_position"] = {p.ProxyToolTarget.Position.X, p.ProxyToolTarget.Position.Y, p.ProxyToolTarget.Position.Z}
        behaviors["base_orientation"] = {p.ProxyToolTarget.Orientation.X, p.ProxyToolTarget.Orientation.Y, p.ProxyToolTarget.Orientation.Z}

        -- Set the starting direction
        local v = p:FindFirstChild("Translate"):GetAttribute("Velocity")
        local startPos = Vector3.new(behaviors["start_position"][1], behaviors["start_position"][2], behaviors["start_position"][3])
        local endPos = Vector3.new(behaviors["end_position"][1], behaviors["end_position"][2], behaviors["end_position"][3])
        behaviors["direction"] = math.sign(v:Dot((endPos - startPos) / behaviors["time"]))
        behaviors["conditions"] = getConditionsFromScript(p, "PersistentTranslate", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "Trampoline") or CollectionService:HasTag(p, "Conveyor") then
        local behaviors = {}
        behaviors["name"] = CollectionService:HasTag(p, "Trampoline") and "Trampoline" or "Conveyor"
        behaviors["velocity"] = HttpService:JSONDecode(p:GetAttribute("velocity"))
        behaviors["conditions"] = getConditionsFromScript(p, behaviors["name"], obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "Waypoint") then
        local behaviors = {}
        behaviors["name"] = "Waypoint"
        behaviors["index"] = p:GetAttribute("index")
        behaviors["jumps"] = p:GetAttribute("jumps")
        behaviors["conditions"] = getConditionsFromScript(p, "Waypoint", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "Avoid") then
        local behaviors = {}
        behaviors["name"] = "Avoid"
        behaviors["conditions"] = getConditionsFromScript(p, "Avoid", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "NoCollide") then
        local behaviors = {}
        behaviors["name"] = "NoCollide"
        behaviors["conditions"] = getConditionsFromScript(p, "NoCollide", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "Unanchored") then
        local behaviors = {}
        behaviors["name"] = "Unanchored"
        behaviors["conditions"] = getConditionsFromScript(p, "Unanchored", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "Disappear") then
        local behaviors = {}
        behaviors["name"] = "Disappear"
        behaviors["disappear_delay"] = p:GetAttribute("disappearDelay")
        behaviors["reappear_delay"] = p:GetAttribute("reappearDelay")
        behaviors["conditions"] = getConditionsFromScript(p, "Disappear", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "TouchTranslate") then
        local behaviors = {}
        behaviors["name"] = "TouchTranslate"

        for i, behaviorAttribute in pairs({"loop", "reverses", "time", "translateType"}) do
            behaviors[behaviorAttribute] = p:GetAttribute(behaviorAttribute)
        end
        behaviors["easing_style"] = p:GetAttribute("easingStyle")
        behaviors["easing_direction"] = p:GetAttribute("easingDirection")

        behaviors["start_position"] = HttpService:JSONDecode(p:GetAttribute("startPosition"))
        behaviors["end_position"] = HttpService:JSONDecode(p:GetAttribute("endPosition"))

        behaviors["base_position"] = {p.ProxyToolTarget.Position.X, p.ProxyToolTarget.Position.Y, p.ProxyToolTarget.Position.Z}
        behaviors["base_orientation"] = {p.ProxyToolTarget.Orientation.X, p.ProxyToolTarget.Orientation.Y, p.ProxyToolTarget.Orientation.Z}

        local v = p:FindFirstChild("Translate"):GetAttribute("Velocity")
        local startPos = Vector3.new(behaviors["start_position"][1], behaviors["start_position"][2], behaviors["start_position"][3])
        local endPos = Vector3.new(behaviors["end_position"][1], behaviors["end_position"][2], behaviors["end_position"][3])
        behaviors["direction"] = math.sign(v:Dot((endPos - startPos) / behaviors["time"]))
        behaviors["conditions"] = getConditionsFromScript(p, "TouchTranslate", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "TouchRotate") then
        local behaviors = {}
        behaviors["name"] = "TouchRotate"
        for i, behaviorAttribute in pairs({"angular", "axis", "loop", "rotateType"}) do
            behaviors[behaviorAttribute] = p:GetAttribute(behaviorAttribute)
        end
        behaviors["base_position"] = {p.ProxyToolTarget.Position.X, p.ProxyToolTarget.Position.Y, p.ProxyToolTarget.Position.Z}
        behaviors["base_orientation"] = {p.ProxyToolTarget.Orientation.X, p.ProxyToolTarget.Orientation.Y, p.ProxyToolTarget.Orientation.Z}
        behaviors["conditions"] = getConditionsFromScript(p, "TouchRotate", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "TouchScale") or CollectionService:HasTag(p, "PersistentScale") then
        local behaviors = {}
        behaviors["name"] = CollectionService:HasTag(p, "TouchScale") and "TouchScale" or "PersistentScale"
        for i, behaviorAttribute in pairs({"loop", "reverses", "time", "scaleType"}) do
            behaviors[behaviorAttribute] = p:GetAttribute(behaviorAttribute)
        end
        behaviors["easing_style"] = p:GetAttribute("easingStyle")
        behaviors["easing_direction"] = p:GetAttribute("easingDirection")
        behaviors["start_scale"] = HttpService:JSONDecode(p:GetAttribute("startScale"))
        behaviors["end_scale"] = HttpService:JSONDecode(p:GetAttribute("endScale"))
        behaviors["conditions"] = getConditionsFromScript(p, behaviors["name"], obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "Friction") then
        local behaviors = {}
        behaviors["name"] = "Friction"
        behaviors["friction"] = p:GetAttribute("friction")
        behaviors["weight"] = p:GetAttribute("frictionWeight")
        behaviors["conditions"] = getConditionsFromScript(p, "Friction", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "HoldState") then
        local behaviors = {}
        behaviors["name"] = "HoldState"
        behaviors["conditions"] = getConditionsFromScript(p, "HoldState", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    if CollectionService:HasTag(p, "PressState") then
        local behaviors = {}
        behaviors["name"] = "PressState"
        behaviors["conditions"] = getConditionsFromScript(p, "PressState", obbyPartNames)
        table.insert(partJson["behaviors"], behaviors)
    end
    -- HACK: Tag procedural model parts with their own behavior that prevents reloading on the backend.
    if CollectionService:HasTag(p, "Procedural") then
        local behaviors = {}
        behaviors["name"] = "Procedural"
        behaviors["conditions"] = {}
        table.insert(partJson["behaviors"], behaviors)
    end

    partJson["idx"] = idx
    return partJson
end

local function createObjectJson(p, idx, sectionTagIndex, debugPositionOffset, obbyJson, obbyPartNames)
    if p:IsA("Model") then
        for i, p_i in p:GetChildren() do
            if p:IsA("Model") or p:IsA("Part") then
                createObjectJson(p_i, idx * 64 + i, sectionTagIndex, debugPositionOffset, obbyJson, obbyPartNames)
            end
            --local partJson = createPartJson(p_i, i, sectionTagIndex, debugPositionOffset)
            --obbyJson["part" .. tostring(idx) .. "_" .. tostring(i)] = partJson
        end
    elseif p:IsA("BasePart") then
        local partJson = createPartJson(p, idx, sectionTagIndex, debugPositionOffset, obbyPartNames)
        if not obbyJson[p.Name] then
            obbyJson[p.Name] = partJson
        else
            -- This is a holdover from the original purpose of this function, which was to jsonify obby sections from real Roblox games.
            obbyJson["part" .. tostring(idx)] = partJson
        end
    end
end

function LoadObbyFunctions.createJsonFromObbySection(sectionTagIndex)
    -- Collect all tagged parts. This tag will be used to infer spawn position and goal location.
    -- Tags are stored as strings, so convert to string for consistency
    local taggedPartsList = CollectionService:GetTagged(tostring(sectionTagIndex))

    partsList = {}
    for _, p in taggedPartsList do
        if p:IsDescendantOf(workspace) then
            table.insert(partsList, p)
        end
    end

    local obbyJson = {}

    -- When exporting, pass the position of the OBBY SPECIFIC starting spawn point so it can be set to world origin.
    -- Makes verifying the export in Madrona with camera flythrough way easier.
    -- For Easy Obby
    -- local debugPositionOffset = Vector3.new(-395.174, 0.65, 318.663)
    -- For Tower of Hell - Easy
    -- local debugPositionOffset = Vector3.new(0.069, 482.595, 22.388)

    debugPositionOffset = Vector3.new(0,0,0)

    -- Build a set of part names in this section for condition lookup
    local obbyPartNames = {}
    for _, p in partsList do
        obbyPartNames[p.Name] = true
    end

    for idx, p in partsList do
        -- Creates JSON entry for this part. Subdivides Models recursively.
        createObjectJson(p, idx, sectionTagIndex, debugPositionOffset, obbyJson, obbyPartNames)
    end
    return obbyJson
end

function LoadObbyFunctions.createObbyFromJson(data, initialAABBs, sectionTagIndex)
    -- Sort according to index. We need this to simplify logic around
    -- ordering objects for agent observation spaces.
    local keys = {}
    for n in pairs(data) do table.insert(keys, n) end
    table.sort(keys, function(a, b)
        return tonumber(data[a]["idx"]) < tonumber(data[b]["idx"])
    end)
    
    -- Optionally pass in a starting tag index
    local tagIdx = Instance.new("IntValue")
    if sectionTagIndex then
        tagIdx.Value = sectionTagIndex
    else
        tagIdx.Value = 1
    end

    -- Parses blocks from obby JSON
    local obbyParts = {}

    -- Create a dummy part to hold states
    local stateFolder = Instance.new("Folder")
    stateFolder.Name = "States"
    -- stateFolder.Parent = workspace -- Put the part into the Workspace
    table.insert(obbyParts, stateFolder)

    for idx, blockPartName in keys do
        table.insert(obbyParts, createBlockPart(blockPartName, data[blockPartName], tagIdx, initialAABBs, stateFolder))
    end

    -- Correct the labels on any loaded waypoints
    adjustWaypointLabels()
    return obbyParts
end

LoadObbyFunctions.adjustWaypointLabels = adjustWaypointLabels

return LoadObbyFunctions